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
import re
import json
import os

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

full_data = text_df.loc[(text_df[mfq_avgs[0]] != -1) & \
                      (text_df[mfq_avgs[1]] != -1) & \
                      (text_df[mfq_avgs[2]] != -1) & \
                      (text_df[mfq_avgs[3]] != -1) & \
                      (text_df[mfq_avgs[4]] != -1)]

http_pattern = re.compile(r"http(s)?[^\s]+")

def sub_link(string, links=dict(), counter=0):
    num_matches = len(http_pattern.findall(string))
    for i in range(num_matches):
        links[counter + i] = http_pattern.search(string).group(0)
        string = http_pattern.sub("link:{}".format(counter + i), string, count=1)
    counter += num_matches
    return string, counter

def extract_web_addresses(df):
    num_links = 0
    link_dict = dict()
    for index, row in df.iterrows():
        text = row["fb_status_msg"]
        new_text, num_links = sub_link(text, link_dict, counter=num_links)
        df.at[index, "fb_status_msg"] = new_text
    return df, link_dict

link_path = source_data_dir + '/' + "web_links.json"
dataframe, link_dict = extract_web_addresses(full_data) # replace http(s):// links with UUIDs
       
if not os.path.isfile(link_path):
    with open(link_path, 'w', encoding='utf-8') as fo:
        json.dump(link_dict, fo, ensure_ascii=False, indent=4)

geo_cols = ["country_current", "county code", "hometown", "zipcode"]
demographics = ["politics" + str(int_val) + "way" for int_val in [10,7,4] ] + \
        ["age", "concentrations", "degrees", "significant_other_id", "relation_status", \
         "religion_now", "politics_new", "sex", "gender"]
demo_race = ["race_" + col for col in ["black", "eastasian", "latino", "middleeast", "nativeamerican", "southasian", "white", "other", "decline"] ]

demographics = geo_cols + demographics + demo_race
predictors = ["fb_status_msg"]
outcomes = [col for col in full_data.columns.values if (col.startswith("MFQ") and col not in ["MFQ_HFminsIAP", "MFQ_BLANKS_perc", "MFQ_failed"]) or col == "mfq_unclear"]

full_selected = dataframe.loc[:, ["userid"] + demographics + predictors + outcomes]

dest = source_data_dir + '/' + "full_dataframe.pkl"
full_selected.to_pickle(dest)

# make concatenated_dataframe
users = dict()
for index, row in full_selected.iterrows():
    userid = row["userid"]
    if userid not in users:
        users[userid] = [row.to_dict()]
    else:
        users[userid].append(row.to_dict())

new_df = list()
for userid in users:
    texts = list()
    for row in users[userid]:
        texts.append(row["fb_status_msg"])
    new_text = "\n".join(texts)
    users[userid][0]["fb_status_msg"] = new_text
    new_df.append(users[userid][0])

concat_df = pd.DataFrame(new_df)
concat_df.to_pickle(source_data_dir + '/' + "concat_df.pkl")

