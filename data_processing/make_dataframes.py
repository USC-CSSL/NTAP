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
mfq_avgs = [col for col in raw_df.columns.values if col.startswith("MFQ") and col.endswith("AVG")]

full_data = raw_df.loc[(raw_df["fb_status_msg"] != -1) & \
                        (raw_df[mfq_avgs[0]] != -1) & \
                        (raw_df[mfq_avgs[1]] != -1) & \
                        (raw_df[mfq_avgs[2]] != -1) & \
                        (raw_df[mfq_avgs[3]] != -1) & \
                        (raw_df[mfq_avgs[4]] != -1)]

print("Adding contraints: all rows must have text & at least one non-null for all foundations\nNew dataframe has {} rows".format(full_data.shape[0]))

print("Cleaning text fields:")
print("Extracting http(s) links")
http_pattern = re.compile(r"http(s)?[^\s]+")

def sub_link(string, links=dict(), counter=0):
    num_matches = len(http_pattern.findall(string))
    for i in range(num_matches):
        links[counter + i] = http_pattern.search(string).group(0)
        string = http_pattern.sub("link{}".format(counter + i), string, count=1)
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

print("Found {} links".format(len(link_dict.values())))

if not os.path.isfile(link_path):
    print("Writing links to {}".format(link_path))
    with open(link_path, 'w', encoding='utf-8') as fo:
        json.dump(link_dict, fo, ensure_ascii=False, indent=4)

print("Cleaning text (transliterate all unicode to ascii-approx) and removing non-English posts")
delete_indices = list()  # for non-English documents
c = 0

for i, row in dataframe.iterrows():
    c += 1
    if c % 1000 == 0:
        print("Processed {} rows".format(c))
        print("{0:.3f}% done".format(100 * float(c) / dataframe.shape[0]))
    text = row["fb_status_msg"]
    decoded_text = unidecode(text).lower()
    collapsed_whitespace = re.sub(r"[\s]+", " ", decoded_text)
    if len(collapsed_whitespace) < 3:
        delete_indices.append(i)  # text is too short
        continue
    detected_language = TextBlob(collapsed_whitespace).detect_language()
    if detected_language != 'en':
        delete_indices.append(i)
    dataframe.at[i, "fb_status_msg"] = collapsed_whitespace

old_rows = dataframe.shape[0]
dataframe = dataframe.drop(delete_indices)
print("Dropped {} rows; new dataframe has {} rows".format(old_rows - dataframe.shape[0], dataframe.shape[0]))

print("Performing lemmatization using WordNetLemmatizer (via nltk)")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

lemmatized_posts = list()
for i, row in dataframe.iterrows():
    text = row["fb_status_msg"].split()
    tags = nltk.pos_tag(text)
    words = list()
    for i in range(len(tags)):
        if len(get_wordnet_pos(tags[i][1])) == 1:
            words.append(lemmatizer.lemmatize(text[i], get_wordnet_pos(tags[i][1])))
        else:
            words.append(lemmatizer.lemmatize(text[i]))
    new_text = " ".join(words)
    lemmatized_posts.append(new_text)
dataframe["lemmatized_posts"] = pd.Series(lemmatized_posts, index=dataframe.index)

print("Added column to dataframe: lemmatized_posts")

geo_cols = ["country_current", "county_code", "hometown", "zipcode"]
demographics = ["politics" + str(int_val) + "way" for int_val in [10,7,4] ] + \
        ["age", "concentrations", "degrees", "significant_other_id", "relation_status", \
         "religion_now", "politics_new", "sex", "gender"]
demo_race = ["race_" + col for col in ["black", "eastasian", "latino", "middleeast", "nativeamerican", "southasian", "white", "other", "decline"] ]

demographics = geo_cols + demographics + demo_race
predictors = ["fb_status_msg", "lemmatized_posts"]
outcomes = [col for col in full_data.columns.values if (col.startswith("MFQ") and col not in ["MFQ_HFminsIAP", "MFQ_BLANKS_perc", "MFQ_failed"]) or col == "mfq_unclear"]

full_selected = dataframe[["userid"] + demographics + predictors + outcomes]
print("Discarding columns with no useful information.\nUpdated dataframe has {} columns\nColumns:{}".format(full_selected.shape[1], '\t'.join(full_selected.columns.tolist())))

dest = source_data_dir + '/' + "full_dataframe.pkl"
full_selected.to_pickle(dest)
print("Wrote full dataframe to {}".format(dest))

# make concatenated_dataframe
print("Making second dataframe with one row per user (concatenate statuses)")
users = dict()
for index, row in full_selected.iterrows():
    userid = row["userid"]
    if userid not in users:
        users[userid] = [row.to_dict()]
    else:
        users[userid].append(row.to_dict())

new_df = list()
for userid in users:
    texts, lemmatized_texts = list(), list()
    for row in users[userid]:
        texts.append(row["fb_status_msg"])
        lemmatized_texts.append(row["lemmatized_posts"])
    new_text = " EOL ".join(texts)
    new_lemmatized = " EOL ".join(lemmatized_texts)
    users[userid][0]["fb_status_msg"] = new_text
    users[userid][0]["lemmatized_posts"] = new_lemmatized
    new_df.append(users[userid][0])

concat_df = pd.DataFrame(new_df)
concat_df.to_pickle(source_data_dir + '/' + "concat_df.pkl")

