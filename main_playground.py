#%%
from lib import airqual_read
from lib import constants
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#%%

# Select geographical location (see ./Data/traffic for list of locations)
#location = "7a41"
location = "1b14"

# Read in air quality data, and downsample it to 1-hour non-overlapping averages to match traffic data.
df = airqual_read.hourly_airqual(location,sensor=0)

# Pick out the second sensor at the defined location.
#df = df_list[1]

# Read in traffic flow data and outdoor air quality data for this location and append to air qual data
traffic_df = pd.read_csv(constants.TRAFFIC_PATH / (location + "_traffic.csv"),index_col=0)

outdoor_df = pd.read_csv(constants.AIR_PATH / "outdoor" / (location + "_outair.csv"),index_col=0)

df = pd.concat([df,traffic_df,outdoor_df],axis=1)

# Two motorways nearby, so we'll aggregate the traffic flows
df["M_count"] = df["M0_count"] + df["M1_count"]

# Correlating variables shows that PM_25 has a weak/moderate relationship with traffic flow
corr = df.corr()

sns.heatmap(corr, cmap="magma",annot=True)



meanval = df["PM_25"].mean() * 1000
print(f"Average PM 2.5 is {np.rint(meanval)} ug/m3. WHO guidelines say aim for average of 5 ug/m3")




