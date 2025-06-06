import tempfile
from concurrent.futures import ThreadPoolExecutor
from clearnights_on_demand.clearnights_at_point import \
    process_clearnights_per_group
import eratos_xarray
import xarray as xr
from datetime import timedelta
import datetime
from dask import delayed
import functools
import dask.array as da
from eratos.adapter import Adapter
from eratos.dsutil.netcdf import gridded_geotime_netcdf_props
import json
import logging
from shapely import wkt
from timezonefinder import TimezoneFinder
from eratos.creds import AccessTokenCreds, BaseCreds
import os
from datetime import datetime
from clearnights.clearnights import ClearNights
import numpy as np

logger = logging.getLogger()
def load_mask_data(
        start_time: str,
        end_time: str,
        geom: str,
        ecreds: AccessTokenCreds,
        clearnights_kwargs: dict
) -> xr.Dataset:
    """
    Load ClearNights data for a given time range and geometry.
    Applies the ClearNights mask to the raw lst data.

    parameters:
    -----------
    start_time: str,
        start time in the format YYYY-MM-DD(optional, if not set, default will be 2025-01-01),
        currently only supports the date in 2025
    end_time: str,
    geom: str, geometry in WKT format (optional, if not set, default will be the whole Australia)
    secret: dict
    returns:
    -----------
    mask: xarray DataArray, ClearNights mask
    raw_lst: xarray DataArray, raw lst data
    masked_lst: xarray DataArray, masked lst data
    """

    try:
        polygon = wkt.loads(geom)
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        print(f"Polygon bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
    except Exception as e:
        raise ValueError("A WKT input is necessary")


    try:
        start_time = datetime.strptime(start_time, "%Y-%m-%d")
        end_time = datetime.strptime(end_time, "%Y-%m-%d")
        print(f"Start date: {start_time}")
        print(f"End date: {end_time}")
    except Exception as e:
        raise ValueError("start_time and end_time must be provided")
    # Convert the string to a datetime object

    preprocessing = lambda x: x.sel(lat=slice(min_lat, max_lat),
                                    lon=slice(min_lon, max_lon),
                                    time=slice(start_time, end_time))

    if not clearnights_kwargs:
        clearnights_kwargs = {}

    clearnights = ClearNights(clearnights_kwargs, tf_kwargs={"in_memory": True})
    start_year = start_time.year
    end_year = end_time.year
    lst_ern = "ern:e-pn.io:resource:csiro.blocks.himawari.lst.2km.24hr.{year}"

    logger.info(
        "Reading in Himawari Satelitte Data for years %d - %d", start_year,
        end_year
    )

    datasets = [
        (
            xr.open_dataset(
                lst_ern.format(year=year), engine="eratos",
                eratos_auth=ecreds
            ).sel(
                lat=slice(min_lat, max_lat),
                lon=slice(min_lon, max_lon),
                time=slice(start_date, end_date),
            )
        )
        for year in range(start_year, end_year + 1)
    ]

    with ThreadPoolExecutor() as executor:
        datasets = list(executor.map(lambda x: x.load(), datasets))
    if len(datasets) > 1:
        dataset = xr.concat(datasets, "time")

    else:
        dataset = datasets[0]

    clearnights = (
        dataset.to_dataframe()
        .rename(columns={"lst": "lst_original"})
        .groupby(["lat", "lon"])
        .apply(lambda x: process_clearnights_per_group(x, clearnights))
    )

    clearnights['lst_stage1_mask'] = clearnights['lst_stage1_mask'].replace(255, np.nan)

    clearnights_gridded = xr.Dataset.from_dataframe(clearnights[["lst_stage1_mask", "night"]])

    mask = clearnights_gridded.pipe(preprocessing).load()
    night = mask['night']
    night_data = night.data.astype(np.uint8)  # shape: (time, lat, lon)
    # Dimensions
    time_len, lat_len, lon_len = night.shape
    # Get the last index with value 1 for each (lat, lon)
    # Mask where night == 1, then multiply by time indices
    time_indices = np.arange(time_len)[:, None, None]  # shape: (time, 1, 1)
    masked_indices = np.where(night_data == 1, time_indices, -1)
    with np.errstate(invalid='ignore'):
        last_1_idx = masked_indices.max(axis=0)  # shape: (lat, lon)

    # Extend the 'night' mask by 3 steps forward in time (if within bounds)
    night_data_extended = night_data.copy()

    def make_mask(t_idx, y_idx, x_idx, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[t_idx, y_idx, x_idx] = 1
        return mask

    delayed_masks = []
    for offset in range(1, 4):  # Extend by 1, 2, 3 steps
        new_indices = last_1_idx + offset
        valid = (new_indices >= 0) & (new_indices < time_len)
        t, y, x = new_indices[valid], *np.where(valid)
        # Delayed creation of the mask
        mask_delayed = delayed(make_mask, name=f"make_mask_{offset}")(
            np.asarray(t), np.asarray(y), np.asarray(x), night.shape)

        # Convert to Dask array lazily
        update_mask = da.from_delayed(mask_delayed, shape=night.shape,
                                      dtype=np.uint8)
        delayed_masks.append(update_mask)

    combined_update_mask = functools.reduce(da.maximum, delayed_masks)
    night_data_extended = da.maximum(night_data_extended,
                                     combined_update_mask)
    print(f"night_data_extended: {night_data_extended.shape}")
    # Create a new DataArray with same coords and dims
    night_extended = xr.DataArray(
        night_data_extended,
        coords=night.coords,
        dims=night.dims,
        name='night_extended'
    )

    print(f"Loading raw lst data from {start_time} to {end_time} in {polygon}")
    raw_lst_slice = dataset.pipe(preprocessing).load() - 273.15
    raw_lst_slice.load().to_netcdf("test/data/raw_lst.nc")
    combined_mask = night_extended * mask['lst_stage1_mask']
    masked_lst = raw_lst_slice.where(combined_mask)

    return masked_lst


def push_to_platform(
        metrics: xr.Dataset,
        fname: str,
        ecreds: AccessTokenCreds,
        geom: str,
        td: str
    ):
    data_fname = f"{fname}.nc"
    data_file_path = os.path.join(td, data_fname)

    metrics.load()
    metrics.to_netcdf(
        data_file_path,
        engine = 'h5netcdf',
        encoding={
            f"{fname}": {"zlib": True, "complevel": 4,
                         "fletcher32": True}
        },
    )
    polygon = wkt.loads(geom)
    eadapter = Adapter(ecreds)
    resource = eadapter.Resource(
        content={
            "@type": "ern:e-pn.io:schema:dataset",
            "type": "ern:e-pn.io:resource:eratos.dataset.type.gridded",
            "name": "Clearnights Gridded Test",
            "description": f"Clearnights data over polygon {polygon} for the period {start_date} to {end_date}",
            "updateSchedule": "ern:e-pn.io:resource:eratos.schedule.noupdate",
            "file": data_fname,
        }
    )

    resource.save()
    filesmap = {data_fname: data_file_path}
    props = gridded_geotime_netcdf_props(filesmap)
    logger.info("Pushing gridded dataset to Eratos")
    resource.data().push_objects(
        "ern::node:au-1.e-gn.io",
        objects=filesmap,
        connector="Objects:Gridded:v1",
        connectorProps=props,
    )
    logger.info(f'{fname}.nc is successfully pushed to Eratos, {resource.ern()}')

    return resource


def daily_frost_metrics(
        geom: str,
        start_date: str,
        end_date: str,
        frost_threshold: float,
        duration_threshold: float,
        ecreds: BaseCreds = None,
):
    f = open("secret.json")
    data = json.load(f)

    eratos_key = data['eratos_key']
    eratos_secret = data["eratos_secret"]

    secret = {'id': eratos_key,
              'secret': eratos_secret}
    if secret is not None:
        ecreds = AccessTokenCreds(**secret)

    if ecreds is None:
        raise ValueError("Creds must be specified")

    masked_lst = load_mask_data(
        start_time=start_date,
        end_time=end_date,
        geom=geom,
        ecreds = ecreds,
        clearnights_kwargs={}
    )

    logger.info("Calculate daily frost metrics")
    if bool(masked_lst['time'].isnull().all()):
        logger.info(
            "Found masked LST which time dimension is None, potentially caused by date mismatch between LST and CN files")
        raise ValueError("Found masked LST which time dimension is None, potentially caused by date mismatch between LST and CN files, please contact admin")

    min_temp = masked_lst.groupby(masked_lst.time.dt.date).min(dim='time')
    min_temp['date'] = min_temp['date'].astype('datetime64[ns]')
    min_temp = min_temp.rename({"lst": "min_temp", 'date': 'time'})

    frost_mask = masked_lst < frost_threshold
    dt_hour = 60 * 10 / 3600
    weighted_frost = np.abs(masked_lst.where(frost_mask)) * dt_hour
    daily_weighted_frost = weighted_frost.groupby(
        weighted_frost.time.dt.date).sum(dim='time')
    daily_weighted_frost['date'] = daily_weighted_frost['date'].astype(
        'datetime64[ns]')
    daily_weighted_frost = daily_weighted_frost.rename(
        {"lst": "frost_hours", 'date': 'time'})

    duration_mask = masked_lst < duration_threshold
    duration = duration_mask.astype(float) * dt_hour
    daily_duration = duration.groupby(duration.time.dt.date).sum(dim='time')
    daily_duration['date'] = daily_duration['date'].astype('datetime64[ns]')
    daily_duration = daily_duration.rename({"lst": "duration", 'date': 'time'})

    logger.info("Creating Eratos resource")

    with tempfile.TemporaryDirectory() as temp_dir:
        min_temp_nc = push_to_platform(metrics=min_temp, fname="min_temp", ecreds=ecreds, geom=geom, td=temp_dir)
        frost_hours_nc = push_to_platform(metrics=daily_weighted_frost, fname="frost_hours", ecreds=ecreds, geom=geom, td=temp_dir)
        duration_nc = push_to_platform(metrics=daily_duration, fname="duration", ecreds=ecreds, geom=geom, td=temp_dir)


        outputs = {
            "frost_metrics_min_temp": min_temp_nc,
            "frost_metrics_frost_hours": frost_hours_nc,
            "frost_metrics_duration": duration_nc
        }


    return outputs


if __name__ == '__main__':
    start_date = "2025-04-01"
    geom = "MULTIPOLYGON (((136.173976025327 -33.0823350440668,136.238068475586 -33.0865101897404,136.23454984744 -33.1405244726167,136.170457397181 -33.1363493269432,136.173976025327 -33.0823350440668)))"
    end_date = "2025-04-10"
    frost_threshold = 0
    duration_threshold = 0
    daily_frost_metrics(
        geom,
        start_date,
        end_date,
        frost_threshold,
        duration_threshold
    )
