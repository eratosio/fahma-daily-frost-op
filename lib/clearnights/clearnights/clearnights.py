import os
import datetime
from datetime import timedelta

import astral
import numpy as np
import pandas as pd
import pytz
import timezonefinder
from astral.sun import sun

path = os.path.abspath(os.path.dirname(__file__))


class ClearNights:

    DEFAULT_SETTINGS = {
        'stage2_air_temp_diff_delta_threshold': 2.0,
        'stage2_type': 'disabled'
    }

    def __init__(self, settings: dict = None, tf_kwargs : dict = None):
        if settings is None:
            settings = {}
        if tf_kwargs is None:
            tf_kwargs = {}
        self._settings = ClearNights.DEFAULT_SETTINGS.copy()
        self._settings.update(settings)
        self.tf = timezonefinder.TimezoneFinder(**tf_kwargs)

    def process_location(self, lst: pd.DataFrame, tair: pd.DataFrame, longitude: float, latitude: float):
        """

        Returns ClearNights filtered version of noctural LST.

        :param lst: pd.DataFrame with a single column containing LST as a float in Kelvin
        :param tair: pd.DataFram with a single column containing air temperature as a float in degrees celcius
        :param longitude:
        :param latitude:
        :return:
        """
        processed_coords = np.array([[longitude, latitude]])
        
        # filter LST values below 250K. these values are artefacts and therefore should be flagged to be excluded from validation
        lst.loc[np.logical_or(lst.lst_original.isnull(), lst.lst_original <= 250), 'artefact'] = 1
        lst.loc[~np.logical_or(lst.lst_original.isnull(), lst.lst_original <= 250), 'artefact'] = 0
        # convert lst to celsius
        lst.loc[lst.lst_original <= 250, 'lst_original'] = np.nan
        lst.lst_original = lst.lst_original.apply(self._kelvin_to_celcius)

        # _stage1_filter uses rolling, which will run into an error if indices are not monotonous
        # shouldn't be an issue with groupby
        lst = lst.sort_index()
        lst['lst_stage1'], lst['lst_stage1_mask'] = self._stage1_filter(lst.lst_original)

        sun_info = self._sun_info(lst.index, processed_coords)

        # Filter out day time values
        
        #lst = lst.loc[sun_info.night]
        lst['night'] = sun_info.night
        lst['lst_stage1_mask'] = lst['lst_stage1_mask'].fillna(False)
        if self._settings.get('stage2_type') != 'disabled':

            if tair is None:
                raise ValueError('tair must be provided to apply stage 2 filter')

            lst['lst_stage2'], lst['lst_stage2_mask'] = self._stage2_filter(lst.lst_stage1, tair)

        lst['night'] = lst['night'].astype('int')
        lst['lst_stage1_mask'] = lst['lst_stage1_mask'].astype('int')
        lst['artefact'] = lst['artefact'].astype('int')

        return lst

    def _kelvin_to_celcius(self, v):
        return v - 273.15

    def _sun_info_for_date(self, date, location_info):

        sun_info = astral.sun.sun(location_info.observer, date=date)

        return pd.Series({'date': date,
                          'sunrise': sun_info['sunrise'],
                          'sunset': sun_info['sunset']
                          })

    def _get_local_daylight_info(self, processed_coords, day_index, approx_tz = True):


        # From the lat/long, get the tz-database-style time zone name (e.g. 'America/Vancouver') or None
        if approx_tz:
            timezone_str = self.tf.timezone_at(lat=processed_coords[0][1], lng=processed_coords[0][0])
        else:
            timezone_str = self.tf.certain_timezone_at(lat=processed_coords[0][1], lng=processed_coords[0][0])

        location_info = astral.LocationInfo('site location', timezone_str.split("/")[0], timezone_str,
                                            processed_coords[0][1],
                                            processed_coords[0][0])
        
        sun_info = pd.Series(day_index).apply(self._sun_info_for_date, location_info=location_info)
   
        return sun_info

    def _sun_info(self, index: pd.Series, processed_coords):
        # Look up sunrise times
        unique_days = np.unique(index.date)

        sun_info = self._get_local_daylight_info(processed_coords, unique_days)

        df = pd.DataFrame({'UTC': index})
        df['date'] = df.UTC.dt.date
        df = pd.merge(df, sun_info, on='date', how='left')
        df.set_index('UTC', inplace=True)

        # Extent night time to one hour after sunrise to account for full extent of potential frosts
        df['night'] = df.apply(lambda r: r['sunset'].time() <= r.name.time() <= (r['sunrise'] + timedelta(hours=1)).time(), axis=1)

        return df

    def _stage1_filter(self, lst_series: pd.Series, window='1h'):

        # Revised version of stage 1
        # Use the absolute values of the first order differences, otherwise too many points are removed
        # Add min_periods = 3 so that any hour with fewer than 3 data points should not output a variance
        lst_stage1_metric = lst_series.diff().abs().rolling(window, min_periods=3, center=True).var()

        lst_stage1_mask = lst_stage1_metric < 0.3
        lst_filtered = lst_series.where(lst_stage1_mask)

        return lst_filtered, lst_stage1_mask

    def _stage2_filter(self, lst_series: pd.Series, tair_series: pd.Series):

        # TODO: Complete implementation and validation of stage 2 filtering
        raise NotImplementedError('Stage 2 filtering not yet implemented')

        #TODO: Where do we validate air temp data. What are the constraints?
        # - hourly or faster
        # - continuous? or some threshold on missing data?
        # - 00:00 alignment?

        # # 1st order difference of air temp
        # # TODO: Question for Ha: Ha's code applies absolute function now rather than after aggregation.
        # tair_series['air_temp_diff'] = tair_series['air_temp'].diff()
        #
        # # Add the air temp to lst (10min) dataframe
        # merged = pd.DataFrame(lst_series).join(tair_series, how='left')
        #
        # # 1st order difference for LST
        # merged['lst_stage1_diff'] = merged['lst_stage1'].diff()
        #
        # # aggregate both 1st order difference streams to hourly, creating hourly frequency for both
        # # TODO: Is this tolerant to air temp not aligned to 00:00
        # # sum() creates the mean rate of change over the hour.
        # # NOTE: reample('h') will align timestamp with start of the hour
        # hourly = merged[['air_temp_diff', 'lst_stage1_diff']].resample('h').agg(pd.Series.sum, skipna=True, min_count=1)
        #
        # # absolute delta between air temp and lst derivatives
        # # Only interested in the absolute difference
        # hourly['air_temp_diff_delta'] = (hourly['air_temp_diff'] - hourly['lst_stage1_diff']).abs()
        #
        # # Thresholding of delta of LST and air temp derivatives
        # delta_threshold = self._settings.get('stage2_air_temp_diff_delta_threshold')
        # hourly['stage2_mask'] = hourly['air_temp_diff_delta'] < delta_threshold # TODO: value for this threshold??
        #
        # # merge mask back with the lst series, stage 2 masks out hourly blocks
        # final_merged = pd.merge_asof(lst_series, hourly['stage2_mask'], left_index=True, right_index=True, direction='forward', tolerance=timedelta(hours=1))
        #
        # lst_stage2 = lst_series.where(final_merged['stage2_mask'])
        #
        # return lst_stage2, final_merged['stage2_mask']
