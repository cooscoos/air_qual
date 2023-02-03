"""Module for making plots"""

import pandas as pd
from matplotlib import pyplot as plt

def overlay_plot(df: pd.DataFrame, var1: str, var1_name: str, var2: str, var2_name: str, colors: list[str,str]):
    """Create overlay time series plots from pandas dataframes
    
    Parameters
    -------
    df: pd.DataFrame
        Input dataframe with index as datetime

    var1, var2: str
        The two dataframe column labels to plot

    var1_name, var2_name: str
        Labels for these variables in the plot


    colors: list[str, str]
        Colors to use in the plot, e.g. ['r','k'] would plot var1 in red and var2 in black

    Outputs
    -------
    Time series plot with var1 and var2 overlayed.
    """

    _,ax = plt.subplots()
    
    # first plot
    df.plot(y=var1,ax=ax,style=f"{colors[0]}-")
    ax.yaxis.label.set_color(colors[0])
    ax.tick_params(axis='y',colors=colors[0])
    ax.set_xlabel("Hour of week")
    ax.set_ylabel(var1_name)

    ax1=ax.twinx()

    # secondary y-axis plot
    df.plot(y=var2,ax=ax1, style=f"{colors[1]}-")
    ax1.set_ylabel(var2_name)
    ax1.yaxis.label.set_color(colors[1])
    ax1.tick_params(axis='y',colors=colors[1])

    # Remove legend
    ax.legend().remove()
    ax1.legend().remove()
