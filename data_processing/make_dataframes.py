"""
@file make_dataframes.py
@author Brendan Kennedy
@date May 9, 2018

argv[1]: path (data_dir)to directory containing facebook/MFQ csv
    - will be destination of any file outputs

Pre-Condition: file exists in data_dir named 'fb_yourmorals_data.csv'
    - file is '|'-separated (vertical bar)
    - To create this file in R:
        > load("fb_ym_full_06-28-17.Rda")
        > write.table(fbDatFull, "fb_yourmorals_data.csv", sep='|')

File purpose: Read in data and create dataframe where:
    - Row has text in it
    - Row has MFQ completed
"""

import pandas as pd
import sys 

"""
def conv_to_dict(my_list):
    d = dict()
    for val in my_list:
        if val not in d:
            d[val] = 1
        else:
            d[val] += 1
    return d
"""

if len(sys.argv) != 2:
    print("Usage: python make_dataframes.py ~/Path/To/Data/")
    exit(1)

source_data_dir = sys.argv[1]
source_data = source_data_dir + '/' + "fb_yourmorals_data.csv"

raw_df = pd.read_csv(source_data, delimiter='|', encoding="latin-1")
raw_df.fillna(value=-1, inplace=True)
mfq_cols = [col for col in raw_df.columns.values if col.startswith("MFQ")]
mfq = mfq_cols[:32]
mfq_avgs = mfq_cols[33:38]

text_df = raw_df.loc[raw_df.fb_status_msg != -1]
no_text_df = raw_df.loc[raw_df.fb_status_msg == -1]

import numpy as np
print(np.mean([val for val in raw_df["MFQ_BLANKS_perc"] if val >=0]))


full_data = text_df.loc[(text_df[mfq_avgs[0]] != -1) | \
                      (text_df[mfq_avgs[1]] != -1) | \
                      (text_df[mfq_avgs[2]] != -1) | \
                      (text_df[mfq_avgs[3]] != -1) | \
                      (text_df[mfq_avgs[4]] != -1)]

dest = source_data_dir + '/' + "full_dataframe.pkl"
print(full_data.shape)
full_data.to_pickle(dest)
