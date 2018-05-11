import pandas as pd
import numpy as np

import sys

if len(sys.argv) != 2:
    print("Usage: python summary_stats.py full_dataframe.py")
    exit(1)

dataframe_file = sys.argv[1]

dataframe = pd.read_pickle(dataframe_file)

for col in dataframe.columns.tolist():
    nulls = len([val for val in dataframe[col].values.tolist() if val == -1]) * 1.
    full_len = len(dataframe[col].values.tolist()) * 1.
    print(nulls / full_len, col)
