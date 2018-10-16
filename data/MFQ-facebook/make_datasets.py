import os
import json

import pandas as pd
import numpy as np


"""
Take original dataframe (all texts, unstructured) and restructure them:
    - multiindex on (user, message)
    - write concatenated post dataframe
    - binarize labels
    - binarize labels (leave out middle)
    - make 3-way classification
"""


def make_multi_index(dataframe):
    """
    Make (author_id, doc_id) multiindex
    """
    list_of_authors = dataframe["userid"].tolist()
    list_of_docids = list(dataframe.index)
    arrays = [list_of_authors, list_of_docids]
    tuples = list(zip(*arrays))
    mindex = pd.MultiIndex.from_tuples(tuples, names=["author_id", "doc_id"])
    dataframe.index = mindex
    return dataframe

def get_meta_data(mindexed_df):
    print("Get user statistics: lengths of posts, etc.")

def split_into_bags(text, num_bags):
    ### Return equal-dist bags of texts
    tokens = text.split()
    partition_size = int(len(tokens) / num_bags)
    if partition_size == 0:
        return [tokens]
    partitions = list()
    for i in range(0, len(tokens), partition_size):
        slice_ = tokens[i: i + partition_size]
        partitions.append(slice_)
    partitions.append(tokens[i: len(tokens)])
    
    return [" ".join(part) for part in partitions]


def get_row(dataframe, userid):
    valid_columns = set(dataframe.columns) - {'userid', 'fb_status_msg'}
    valid_columns = list(valid_columns)
    subset = dataframe.loc[[int(userid),], valid_columns]
    return subset.iloc[0]

def make_concatenated(mindexed_df, num_bags=10, threshold=5):
    author_ids, doc_ids = zip(*(list(mindexed_df.index)))
    authors_uniq = list(set(author_ids))
    bag_dict = list()
    concat_dict = list()
    for auth in authors_uniq:
        text_series = mindexed_df.loc[auth, "fb_status_msg"]
        concat = " ".join(text_series.tolist())
        if len(text_series) >= threshold:
            ### Leaving out sparse users; those who have less bags than threshold
            rebagged = split_into_bags(concat, num_bags)    
            rest_of_row = get_row(mindexed_df, auth)
            for bag in rebagged:
                new_bagged_row = {"userid": auth, "fb_status_msg": bag}
                for key, value in rest_of_row.items():
                    new_bagged_row[key] = value
                bag_dict.append(new_bagged_row)
            new_concat_row = {"userid": auth, "fb_status_msg": concat}
            for k, v in rest_of_row.items():
                new_concat_row[k] = v
            concat_dict.append(new_concat_row)
    concat_df = pd.DataFrame(concat_dict)
    concat_df.index = concat_df["userid"]
    rebagged_df = pd.DataFrame(bag_dict)
    rebagged_df.index = rebagged_df["userid"]

    return concat_df, rebagged_df


def discretize(dataframe, columns, option):
    for col in columns:
        missing = dataframe.loc[dataframe[col] == -1.].index
        nonnull = dataframe.loc[dataframe[col] != -1.].index
        vector = dataframe[col].drop(missing).values
        disc_vec = pd.qcut(vector, 4, labels=False)
        dataframe.loc[ nonnull, col] = disc_vec

if __name__ == '__main__':
    source_path = os.environ['SOURCE_PATH']
    dataset_path = os.path.dirname(source_path)
    params_path = os.environ['PARAMS']
    with open(params_path, 'r') as fo:
        params = json.load(fo)
    source_df = pd.read_pickle(source_path)

    if params["discretize"] is not None:
        discretize(source_df, params['target_cols'], params["discretize"])
    make_multi_index(source_df)
    concat, rebagged = make_concatenated(source_df)

    ### Write data, based on bagging options
    bagging_option = params["group_by"]
    dataset_path = os.path.join(dataset_path, bagging_option + '.pkl')
    os.environ['DATASET_PATH'] = dataset_path  # has (at least) the text we're interested in, structured by spec.
   

    if bagging_option == 'user':
        concat.to_pickle(dataset_path)
    elif bagging_option == 'user-bagged':
        rebagged.to_pickle(dataset_path)
    else:
        source_df.to_pickle(dataset_path)
