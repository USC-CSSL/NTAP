"""
@file make_dataframes.py
@author Brendan Kennedy
@startdate May 9, 2018
@endate May 18, 2018

argv[1]: path (data_dir)to directory containing facebook/MFQ csv
    - will be destination of any file outputs

Pre-Condition: file exists in data_dir named 'fb_yourmorals_data.csv'
    - file is '|'-separated (vertical bar)
    - To create this file in R:
        > load("fb_ym_full_06-28-17.Rda")
        > data <- data.frame(lapply(fbDatFull, function(x) {
            gsub("\\|", ";", x)
            }))
        > write.table(data, "fb_yourmorals_data.csv", sep='|')

File purpose: Read in data and create dataframe where:
    - Row has text in it
    - Row has MFQ completed (at least one answer per foundation)
Additional Tasks:
    - drop non-English texts
    - select appropriate columns (demographics, text, MFQ outcomes)
    - convert unicode to ascii
    - perform lemmatization (using WordNet)
    - output two dataframes: individual message and concatenated messages
"""

import pandas as pd
import sys 
import re
import json
import os
from unidecode import unidecode
from nltk.corpus import wordnet
import nltk
from textblob import TextBlob

if len(sys.argv) != 2:
    print("Usage: python make_dataframes.py ~/Path/To/Data/")
    exit(1)

source_data_dir = sys.argv[1]
source_data = source_data_dir + '/' + "fb_yourmorals_data.csv"

raw_df = pd.read_csv(source_data, delimiter='|', encoding="latin-1", low_memory=False)
print("Extracting dataframe from {0:s} with {1} rows and {2} columns".format(source_data, raw_df.shape[0], raw_df.shape[1]))

raw_df.fillna(value=-1, inplace=True)
# mfq_avgs = [col for col in raw_df.columns.values if col.startswith("MFQ") and col.endswith("AVG")]
full_data = raw_df.loc[(raw_df["fb_status_msg"] != -1)]

print("Adding contraints: all rows must have text")
print("Removed {} rows".format(len(raw_df) - len(full_data)))
geo_cols = ["country_current", "county_code", "hometown", "zipcode"]
demographics = ["politics" + str(int_val) + "way" for int_val in [10,7,4] ] + \
        ["age", "concentrations", "degrees", "significant_other_id", "relation_status", \
         "religion_now", "politics_new", "sex", "gender"]
demo_race = ["race_" + col for col in ["black", "eastasian", "latino", "middleeast", "nativeamerican", "southasian", "white", "other", "decline"] ]

demographics = geo_cols + demographics + demo_race
outcomes = [col for col in full_data.columns.values if (col.startswith("MFQ") and col not in ["MFQ_HFminusIAP", "MFQ_BLANKS_perc", "MFQ_failed"]) or col == "mfq_unclear"]

full_selected = full_data[["userid", "fb_status_msg"] + demographics + outcomes]
print("Discarding columns with no useful information.\nUpdated dataframe has {} columns".format(full_selected.shape[1]))

# make concatenated_dataframe
print("Making dataframe with one row per user (concatenate statuses)")
users = dict()
for index, row in full_selected.iterrows():
    userid = row["userid"]
    if userid not in users:
        users[userid] = [row.to_dict()]
    else:
        users[userid].append(row.to_dict())

new_data = list()
for userid in users:
    texts = list()
    for row in users[userid]:
        texts.append(row["fb_status_msg"])
    users[userid][0]["fb_status_msg"] = texts
    new_data.append(users[userid][0])

df = pd.DataFrame(new_data)
df.to_pickle(source_data_dir + '/' + "yourmorals_df.pkl")
