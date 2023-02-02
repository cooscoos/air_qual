"""Module for converting air quality csvs into hourly-averaged pandas df"""

from . import constants
import pandas as pd

def hourly_airqual(location: str, lag: int) -> list[pd.DataFrame]:
    """Reads in air quality csvs from given location, takes non-overlapping 1-hour averages of readings, returns a list of dataframes.
    
    Parameters
    -------
    location: str
        A geographical location code that matches filename. See ./README.md
    

    Returns
    -------
    df_list: list[pd.DataFrame]
        Dataframes of hourly averaged air quality readings.
        One df per sensor platform at the given location.
    """
    
    airqual_path = constants.AIR_PATH / "indoor" / location

    df_list = []
    for sensor_plat in airqual_path.glob('*'):
        
        # Gather the data for this sensor platform in df
        df = pd.DataFrame()
        for file in sensor_plat.glob('*'):
            temp_df = pd.read_csv(file,skiprows=range(1,4)) # skip header and first time-point (always garbage)
            
            # remove timezone info, could be key to getting UK time later
            temp_df.set_index(pd.to_datetime(temp_df["TIME"]).dt.tz_convert(None),inplace=True)
            temp_df.drop(["TIME"],axis=1,inplace=True) # drop the unneeded time column
            df = pd.concat([df,temp_df])        
            
        # Need to knock an hour off because sensors are set to Europe/Madrid time, which at time of data collection is 1 hour ahead
        df.index = df.index - pd.Timedelta(hours=lag)
        df.sort_index() # sort by time to be safe
        df.index.names = ["DateTime"]

        # Expand DateTime to day of week and hour of day
        df["day"] = df.index.day_of_week
        df["hour"] = pd.to_timedelta(df.index.hour,unit='H')


        # Slice out each day of the week and take 1-hour, non-overlapping averages of readings.
        # Then, stitch all days of the week together
        stitched_df = pd.DataFrame()
        for day in range(0,7):
            avg_df = df[df["day"]==day].groupby(df["hour"]).mean(numeric_only=True)
            stitched_df = pd.concat([stitched_df,avg_df])          

        # Reset the index so that it becomes "hour of the week" rather than "hour of the day"
        stitched_df.reset_index(inplace=True,drop=True)
    
        df_list.append(stitched_df)

    return df_list