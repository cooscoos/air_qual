#%%
from lib import airqual_read
from lib import constants
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# First let's read in the air quality data, then downsample it to 1 hour non-overlapping bins just like the traffic data

#location = "7a41"
location = "1b14"
lag = -1

df_list = airqual_read.hourly_airqual(location,lag)

# Pick out the second sensor at this location
df = df_list[1]

# Read in traffic flow data for this location and append to air qual data
traffic_df = pd.read_csv(constants.TRAFFIC_PATH / (location + "_traffic.csv"),index_col=0)
df = pd.concat([df,traffic_df],axis=1)

# Two motorways nearby, so we'll aggregate the traffic flows
df["M_count"] = df["M0_count"] + df["M1_count"]

# Correlate variables shows that PM_25 has a weak/moderate relationship with traffic flow
df.corr()

# Let's plot the time series
fig,ax = plt.subplots()
df.plot(y="M_count",ax=ax,style="k-")
ax.set_xlabel("Hour of week")
ax.set_ylabel("Motorway traffic flow (vehicles)")
ax1=ax.twinx()
df.plot(y="PM_25",ax=ax1, style="r-")
ax1.set_ylabel("PM 2.5 (ppm)")
ax1.yaxis.label.set_color("red")
ax1.tick_params(axis='y',colors="red")
ax.legend().remove()
ax1.legend().remove()

# And scatter
df.plot.scatter(x="M_count",y="PM_25", ylabel="PM 2.5 (ppm)",xlabel="Motorway traffic flow (vehicles)")

meanval = df["PM_25"].mean() * 1000
print(f"Average PM 2.5 is {np.rint(meanval)} ug/m3. WHO guidelines say aim for average of 5 ug/m3")


# %%
