"""
Warning: Deprecated file (not in use anymore). 
Just keeping it here in case I need it someday
"""

import pandas as pd
import json
import numpy as np
import os
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python summary_stats.py Path/To/full_dataframe.pkl [Path/To/users.json]")
    exit(1)

dataframe_file = sys.argv[1]
users_file = sys.argv[2]

dataframe = pd.read_pickle(dataframe_file)

def build_user_dict(df):
    users = dict()

    for index, row in df.iterrows():
        user_id = str(row["userid"])
        message = row["fb_status_msg"]
        if user_id not in users:
            users[user_id] = [message]
        else:
            users[user_id].append(message)
    return users

if not os.path.isfile(users_file):
    user_dict = build_user_dict(dataframe)
    with open(users_file, 'w', encoding='utf-8') as fo:
        json.dump(user_dict, fo, indent=4, ensure_ascii=False)
else:
    with open(users_file, 'r', encoding='utf-8') as fo:
        user_dict = json.load(fo)

sizes = [len(user_dict[key]) for key in user_dict]
avg_lengths = [np.mean([len(message.split()) for message in user_dict[user]]) for user in user_dict]

num_users = len(user_dict)

print("Dataframe has {0} unique users with {1} total messages\n\
        Min: {2}\n\tMedian: {3}\n\tMean: {4:0.3f}\n\tMax: {5}".format(num_users,\
        sum(sizes), min(sizes), np.median(sizes), np.mean(sizes), max(sizes)))

print()
print("Lengths of sentences (aggregated per user):\n\tMin: {0}\n\tMean: {1:0.2f}\n\tMax: {2}".format(min(avg_lengths), np.mean(avg_lengths), max(avg_lengths)))
