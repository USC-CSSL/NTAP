import pandas as pd
import numpy as np

def analyze_targets(df, target_cols):
    copied = df.copy()
    copied[copied < 0] = np.nan
    data = copied[target_cols]
    covs = data.cov().values
    for row in zip(target_cols, covs.diagonal()):
        print(row)
