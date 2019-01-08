import re, os, json, nltk, string, emot, emoji, sys
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
import argparse

from parameters import processing

def link(df, col):
    print("Extracting http(s) links")
    http_pattern = re.compile(r"http(s)?[^\s]+")
    texts = df[col].values.tolist()
    links = [http_pattern.findall(text) for text in texts]
    new_texts = [http_pattern.sub("", text) for text in texts]
    df.loc[:, col] = pd.Series(new_texts, index=df.index)
    df.loc[:, "links"] = pd.Series(links, index=df.index)
    return df

def hashtag(df, col):
    print("Extracting hashtags")
    hashtag_pattern = re.compile(r"#[a-zA-Z0-9]+")
    texts = df[col].values.tolist()
    hashtags = [hashtag_pattern.findall(text) for text in texts]
    new_texts = [hashtag_pattern.sub("", text) for text in texts]
    df.loc[:, col] = pd.Series(new_texts, index=df.index)
    df.loc[:, "hashtags"] = pd.Series(hashtags, index=df.index)
    return df


def mentions(df, col):
    print("Extracting mentions")
    mention_re = re.compile(r"@[a-zA-Z0-9]+")
    texts = df[col].values.tolist()
    mentions = [mention_re.findall(text) for text in texts]
    new_texts = [mention_re.sub("", text) for text in texts]
    df.loc[:, col] = pd.Series(new_texts, index=df.index)
    df.loc[:, "mentions"] = pd.Series(mentions, index=df.index)
    return df

def emojis(df, col):
    emojis_col = list()
    for i, row in df.iterrows():
        text = row[col]

        emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
        emoji_pattern = '|'.join(re.escape(p) for p in emojis_list)
        r = re.compile(emoji_pattern)
        this_emojis = r.findall(text)
        text = re.sub(emoji_pattern, "", text)

        for emo in emot.emoticons(text):
            if len(emo['value']) > 1:
                this_emojis.append(emo['value'])
                text = text.replace(emo['value'], "")
        emojis_col.append(this_emojis)
        df.at[i, col] = text
    print("Removed all emojis from the text.")
    print("Added emojis as a new column to the dataframe")
    df["emojis"] = pd.Series(emojis_col, index=df.index)
    return df

def stem(df, col):
    print("Stemming with Porter Stemmer")
    stemmer = PorterStemmer()
    texts = df[col].values.tolist()
    new_texts = [" ".join([stemmer.stem(w) for w in text.split()]) for text in texts]
    df.loc[:, col] = new_texts
    return df

"""
def english(df, col):
    print("Removing non-English posts")
    delete_indices = list()  # for non-English documents
    c = 0

    for i, row in df.iterrows():
        c += 1
        if c % 1000 == 0:
            print("Processed {} rows".format(c))
            print("{0:.3f}% done".format(100 * float(c) / df.shape[0]))
        text = row[col]
        detected_language = TextBlob(text).detect_language()
        if len(text) < 3:
            delete_indices.append(i)  # text is too short
            continue
        if detected_language != 'en':
            delete_indices.append(i)
    old_rows = df.shape[0]
    df = df.drop(delete_indices)
    print("Dropped {} rows; new dataframe has {} rows".format(old_rows - df.shape[0], df.shape[0]))
    return df

"""

def parse_input(file_str):

    ending = file_str.split('.')[-1]
    if ending == 'pkl':
        dataframe = pd.read_pickle(file_str)
    if ending == 'csv':
        dataframe = pd.read_csv(file_str, delimiter=',')
    if ending == 'tsv':
        dataframe = pd.read_csv(file_str, delimiter='\t')
    # add more options for inputting data

    return dataframe


def select_text_col(df):
    cols = df.columns.tolist()
    print("Dataframe has {} columns:".format(len(cols)))
    for col in cols:
        print("\t{}".format(col))
    notvalid = True
    while notvalid:
        text_col = input("Enter text col from those above: ")
        if text_col.strip() not in cols:
            print("Not a valid column name")
        else:
            notvalid = False
    return text_col

def select_target_col(df):
    cols = df.columns.tolist()
    notvalid = True
    while notvalid:
        target_col = input("Enter target col from those above: ")
        if target_col.strip() not in cols:
            print("Not a valid column name")
        else:
            notvalid = False
    return target_col

def lowercase(df, text_col):
    texts = [text.lower() for text in df[text_col].values]
    df[:, text_col] = pd.Series(texts, index=df.index)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to source file")
    parser.add_argument("--output", help="Path to file in which to store processed data")
    args = parser.parse_args()

    indata = parse_input(args.input)

    text_col = select_text_col(indata)
    target_col = select_target_col(indata)

    extract = processing["extract"]
    preprocess = processing["preprocess"]
    lower = processing["lower"]

    dataframe = indata.loc[:, [text_col, target_col] ]

    for method in extract:
        globals()[method](dataframe, text_col)    
    for method in preprocess:
        globals()[method](dataframe, text_col)
    
    dataframe.rename(mapper={text_col: "text", target_col: "target"}, 
                     axis='columns', inplace=True)
    
    dataframe.to_pickle(args.output)
