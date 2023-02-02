#%%
from lib import constants
import pandas as pd
import re
# First let's read in the air quality data, then downsample it to 1 hour non-overlapping bins just like the traffic data

#location = "7a41"
location = "1b14"

pathy = constants.AIR_PATH / "indoor" / location

#start_date = "22-10-01"
#end_date = "22-10-26"

for sublocation in pathy.glob('*'):
    df = pd.DataFrame()
    # Each directory is a room
    print(sublocation)
    #read_in = False

    for file in sublocation.glob('*'):

#        if re.search(start_date,file.as_posix):
#            read_in = True
#        elif re.search(end_date,file.as_posix):
#            read_in = False

#        if read_in:        
            print(file)
            tempo = pd.read_csv(file,skiprows=range(1,4)) # skip header and first time-point (always garbage)
            # remove timezone info, could be key to getting UK time later
            tempo.set_index(pd.to_datetime(tempo["TIME"]).dt.tz_convert(None),inplace=True)
            tempo.drop(["TIME"],axis=1,inplace=True) # drop the additional time column
            df = pd.concat([df,tempo])        
        
    df.sort_index()

    df.plot(y="SCD30_CO2")  
# First job is to dir in here and stitch the data


# %%
