"""This contains project constants"""

from pathlib import Path    # path mgmt

# Relative path main data dir
DATA_PATH = Path("Data")

# Relative paths for traffic and air_quality data
TRAFFIC_PATH = DATA_PATH / "traffic"
AIR_PATH = DATA_PATH / "air_qual"

# Relative paths for outputs
OUTPUT_PATH = Path("Output")