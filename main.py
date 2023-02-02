#%%
from lib import airqual_read
from lib import constants
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# First let's read in the air quality data, then downsample it to 1 hour non-overlapping bins just like the traffic data

#location = "7a41"
location = "1b14"

lag = 1

df_list = airqual_read.hourly_airqual(location,lag)


df = df_list[1]

# Read in traffic flow data as a df
traffic_path = constants.TRAFFIC_PATH / (location + "_traffic.csv")

df2 = pd.read_csv(traffic_path,index_col=0)

data = pd.concat([df,df2],axis=1)

fig,ax = plt.subplots()
#data.plot(y="SCD30_CO2",ax=ax, style="r-")
data.plot(y="PM_25",ax=ax, style="r-")
ax1=ax.twinx()
data.plot(y="M0_count",ax=ax1,style="k-")


# let's scatter
data.plot.scatter(x="M0_count",y="PM_25")

data.corr()
# %%
