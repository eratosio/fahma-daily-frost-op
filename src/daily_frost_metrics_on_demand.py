import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
import h5netcdf
import eratos_xarray
import xarray as xr
from shapely.geometry import Polygon, MultiPolygon
from dask import delayed
import functools
import dask.array as da
from eratos.adapter import Adapter
from eratos.dsutil.netcdf import gridded_geotime_netcdf_props
import json
import logging
from shapely import wkt
from eratos.creds import AccessTokenCreds, BaseCreds
from eratos_xarray.backend.eratos_ import EratosDataStore
import os
from pyproj import Geod
from datetime import datetime, timedelta
from clearnights.clearnights import ClearNights
from clearnights.xarray import process_clearnights_xarray
from shapely.errors import WKTReadingError
import numpy as np

logger = logging.getLogger()

def load_himawari_datasets(
    start_date: str,
    end_date: str,
    ecreds: BaseCreds,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
) -> xr.Dataset:
    lst_ern = "ern:e-pn.io:resource:csiro.blocks.himawari.lst.2km.24hr.{year}"
    start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    end_year = datetime.strptime(end_date, "%Y-%m-%d").year

    logger.info(
        "Reading in Himawari Satellite Data for years %d - %d", start_year, end_year
    )


    logger.info(
        f"Slicing data to bbox lat: ({lat_min}, {lat_max}), lon: ({lon_min}, {lon_max}) time: {start_date} - {end_date}"
    )

    datasets = [
        (
            xr.open_dataset(
                lst_ern.format(year=year), engine="eratos", eratos_auth=ecreds
            )
            .sortby('time')
            .sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
                time=slice(start_date, end_date),
            )
        )
        for year in range(start_year, end_year + 1)
    ]

    with ThreadPoolExecutor() as executor:
        datasets = list(executor.map(lambda x: x.load(), datasets))

    return xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]

def load_dataset_for_year(year, lst_ern, adapter, preprocessing):
    resource = adapter.Resource(lst_ern.format(year=year))
    data = resource.data()
    gapi = data.gapi()
    store = EratosDataStore(gapi)
    return xr.open_dataset(store).pipe(preprocessing)

def load_mask_data(
        start_time: str,
        end_time: str,
        ecreds: AccessTokenCreds,
        polygon: Polygon|MultiPolygon,
        eadapter: Adapter,
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
    polygon: Polygon or MultiPolygon,
    eadapter: Adapter
    returns:
    -----------
    mask: xarray DataArray, ClearNights mask
    raw_lst: xarray DataArray, raw lst data
    masked_lst: xarray DataArray, masked lst data
    """

    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    print(f"Polygon bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")


    if not clearnights_kwargs:
        clearnights_kwargs = {}

    clearnights = ClearNights(clearnights_kwargs, tf_kwargs={"in_memory": True})
    himawari_data = load_himawari_datasets(
        start_time,
        end_time,
        ecreds,
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
    )

    clearnights_gridded = process_clearnights_xarray(
        himawari_data, clearnights
    )

    mask = clearnights_gridded.load()
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
    KELVIN_TO_CELSIUS = -273.15
    raw_lst_slice = himawari_data.load() + KELVIN_TO_CELSIUS
    night_extended = night_extended.where(night_extended == 1, np.nan)
    mask['lst_stage1_mask'] = mask['lst_stage1_mask'].where(mask['lst_stage1_mask'] == 1, np.nan)
    combined_mask = night_extended * mask['lst_stage1_mask']
    masked_lst = raw_lst_slice.where(combined_mask)

    return masked_lst


def push_to_platform(
        metrics: xr.Dataset,
        fname: str,
        eadapter: Adapter,
        polygon: Polygon|MultiPolygon,
        td: str,
        start_date: str,
        end_date: str
    ):
    data_fname = f"{fname}.nc"
    data_file_path = os.path.join(td, data_fname)

    metrics.load()
    metrics.to_netcdf(
        data_file_path,
        engine="h5netcdf",
        encoding={
            f"{fname}": {"zlib": True, "complevel": 4,
                         "fletcher32": True, "_FillValue": np.finfo(np.float32).max}
        },
    )
    resource = eadapter.Resource(
        content={
            "@type": "ern:e-pn.io:schema:dataset",
            "type": "ern:e-pn.io:resource:eratos.dataset.type.gridded",
            "name": f"frost_metrics_{fname}",
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
    logger.info(
        "Resulting resource manifest - \n %s",
        json.dumps(json.loads(resource.json()), indent=4),
    )

    return resource


def daily_frost_metrics(
        geom: str,
        start_date: str,
        end_date: str,
        frost_threshold: float,
        duration_threshold: float,
        ecreds: BaseCreds = None,
        secret: Dict[str, str] = None,
):

    # Validate the input parameters

    if secret is not None:
        ecreds = AccessTokenCreds(**secret)

    if ecreds is None:
        raise ValueError("Creds must be specified")

    eadapter = Adapter(ecreds)

    try:
        polygon = wkt.loads(geom)

    except WKTReadingError as e:
        raise ValueError(f"Invalid WKT geometry: {geom}") from e
    except AttributeError as e:
        raise ValueError(
            "Input must be a valid WKT string representing a polygon") from e


    if type(polygon) is Polygon:
        polygonList = [polygon]
    elif type(polygon) is MultiPolygon:
        polygonList = list(polygon.geoms)
    else:
        raise ValueError(
            f'geom must be either a polygon or multiple polygons (found {polygon.geom_type})')

    # Check we aren't processing more than 2000 sq.km .
    geod = Geod(ellps="WGS84")
    totalAreaKM = 0.0
    for poly in polygonList:
        totalAreaKM += abs(geod.geometry_area_perimeter(poly)[0]) / 1000000.0
    print("Input polygon size: ", totalAreaKM)
    if totalAreaKM > 2000:
        print(f'total area of geometry {totalAreaKM} exceeds 2000 sq.km')
        raise ValueError(
            f'total area of geometry {totalAreaKM} exeeds 2000 sq.km, please enter a smaller area')

    try:
        start_time = datetime.strptime(start_date, "%Y-%m-%d")
        end_time = datetime.strptime(end_date, "%Y-%m-%d")
        print(f"Start date: {start_time}")
        print(f"End date: {end_time}")
    except ValueError as e:
        raise ValueError("start_time and end_time must be provided")
    if end_time - start_time > timedelta(days=2 * 365):
        raise ValueError("Date span should not be exceed 2 years")

    # Calculate Clearnight mask
    masked_lst = load_mask_data(
        start_time=start_date,
        end_time=end_date,
        ecreds=ecreds,
        polygon=polygon,
        eadapter = eadapter,
        clearnights_kwargs={}
    )

    logger.info("Calculate daily frost metrics")
    if bool(masked_lst['time'].isnull().all()):
        logger.info(
            "Found masked LST which time dimension is None, potentially caused by date mismatch between LST and CN files")
        raise ValueError("Found masked LST which time dimension is None, "
                         "potentially caused by date mismatch between LST and CN files, "
                         "please contact support@eratos.com")

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

        min_temp_nc = push_to_platform(metrics=min_temp, fname="min_temp", eadapter=eadapter, polygon=polygon, td=temp_dir, start_date=start_date, end_date=end_date)
        frost_hours_nc = push_to_platform(metrics=daily_weighted_frost, fname="frost_hours", eadapter=eadapter, polygon=polygon, td=temp_dir, start_date=start_date, end_date=end_date)
        duration_nc = push_to_platform(metrics=daily_duration, fname="duration", eadapter=eadapter, polygon=polygon, td=temp_dir, start_date=start_date, end_date=end_date)


        outputs = {
            "frost_metrics_min_temp": min_temp_nc,
            "frost_metrics_frost_hours": frost_hours_nc,
            "frost_metrics_duration": duration_nc
        }

        return outputs


# if __name__ == '__main__':
#     frost_threshold_lst = [0, 1]
#     duration_threshold_lst = [0, 1]
#     start_year = 2025
#
#     f = open("secret.json")
#     data = json.load(f)
#
#     eratos_key = data['eratos_key']
#     eratos_secret = data["eratos_secret"]
#
#     secret = {'id': eratos_key,
#               'secret': eratos_secret}
#
#     geom = "POLYGON ((115.977173 -31.905541, 116.356201 -31.905541, 116.356201 -31.688445, 115.977173 -31.688445, 115.977173 -31.905541))"
#
#     daily_frost_metrics(
#         geom=geom,
#         duration_threshold=1,
#         frost_threshold = 1,
#         start_date ="2025-08-05",
#         end_date = "2025-08-06",
#         secret=secret
#     )
#

