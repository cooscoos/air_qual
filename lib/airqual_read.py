"""Module for converting air quality csvs into hourly-averaged pandas df"""

from . import constants
import pandas as pd
import re

def hourly_airqual(location: str, sensor: int = 0, lag: int = -1) -> pd.DataFrame:
    """Reads in air quality csvs from given location and sensor, takes non-overlapping 1-hour averages of readings, returns a dataframe.
    
    Parameters
    -------
    location: str
        A geographical location code that matches filename. See ./README.md

    sensor: int <Optional, default=0>
        When there are multiple sensors at the location, choose a sensor number 0, 1, 2, ...

    lag: int <Optional, default=-1>
        Apply lag of n hours to the time series. Useful when trying to correlate air quality
        with other data. Default brings the time series to the correct UTC time (as sensors
        programmed to report Europe/Madrid time).

    Returns
    -------
    stitched_df: pd.DataFrame
        Dataframe of hourly averaged air quality readings for given sensor at location.
    """
    
    airqual_path = constants.AIR_PATH / "indoor" / location

    stitched_df = pd.DataFrame()
    for sensor_plat in airqual_path.glob('*'):
        if sensor_plat.stem == f"{sensor}":
            # Gather the data for this sensor platform in df
            df = pd.DataFrame()
            for file in sensor_plat.glob('*'):
                temp_df = pd.read_csv(file,skiprows=range(1,4)) # skip header and first time-point (always garbage)
                
                # remove timezone info, could be key to getting UK time later
                temp_df.set_index(pd.to_datetime(temp_df["TIME"]).dt.tz_convert(None),inplace=True)
                temp_df.drop(["TIME"],axis=1,inplace=True) # drop the unneeded time column
                df = pd.concat([df,temp_df])        
                
            # Need to knock an hour off because sensors are set to Europe/Madrid time, which at time of data collection is 1 hour ahead
            df.index = df.index + pd.Timedelta(hours=lag)
            df.sort_index() # sort by time to be safe
            df.index.names = ["DateTime"]

            # Expand DateTime to day of week and hour of day
            df["day"] = df.index.day_of_week
            df["hour"] = pd.to_timedelta(df.index.hour,unit='H')


            # Slice out each day of the week and take 1-hour, non-overlapping averages of readings.
            # Then, stitch all days of the week together

            for day in range(0,7):
                avg_df = df[df["day"]==day].groupby(df["hour"]).mean(numeric_only=True)
                stitched_df = pd.concat([stitched_df,avg_df])          

            # Reset the index so that it becomes "hour of the week" rather than "hour of the day"
            stitched_df.reset_index(inplace=True,drop=True)
        
    return stitched_df

