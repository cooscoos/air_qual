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
data["M_count"] = data["M0_count"] + data["M1_count"]

fig,ax = plt.subplots()
data.plot(y="M_count",ax=ax,style="k-")
ax.set_xlabel("Hour of week")
ax.set_ylabel("Motorway traffic flow (vehicles)")
ax1=ax.twinx()
data.plot(y="PM_25",ax=ax1, style="r-")
ax1.set_ylabel("PM 2.5 (ppm)")
ax1.yaxis.label.set_color("red")
ax1.tick_params(axis='y',colors="red")
ax.legend().remove()
ax1.legend().remove()

# let's scatter
data.plot.scatter(x="M_count",y="PM_25", ylabel="PM 2.5 (ppm)",xlabel="Motorway traffic flow (vehicles)")

meanval = data["PM_25"].mean() * 1000
print(f"Average PM 2.5 is {np.rint(meanval)} ug/m3. WHO guidelines say aim for average of 5 ug/m3")

data.corr()
# %%
