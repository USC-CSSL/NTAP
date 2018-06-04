import re, os, json, nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize
from unidecode import unidecode
from textblob import TextBlob


def preprocess_text(df,
                    col,
                    methods,
                    data_dir ):
    pre_list = ["lemmatize", "link", "hashtag", "stopwords", ]

    if not set(methods).issubset(pre_list):
        print("Some preprocessing methods are not available")
        exit(1)

    if col not in df.columns:
        print("{} is not a column name".format(col))
        exit(1)

    for method in methods:
        df = globals()[method](df, col, data_dir)


def link(df, col, data_dir):
    print("Extracting http(s) links")
    num_links = 0
    link_dict = dict()
    for index, row in df.iterrows():
        text = row[col]
        new_text, num_links = sub_link(text, link_dict, counter=num_links)
        df.at[index, col] = new_text
    link_path = data_dir + '/' + "web_links.json"
    if not os.path.isfile(link_path):
        print("Writing links to {}".format(link_path))
        with open(link_path, 'w', encoding='utf-8') as fo:
            json.dump(link_dict, fo, ensure_ascii=False, indent=4)
    return df


def sub_link(string, links, counter=0):
    http_pattern = re.compile(r"http(s)?[^\s]+")
    num_matches = len(http_pattern.findall(string))
    for i in range(num_matches):
        links[counter + i] = http_pattern.search(string).group(0)
        string = http_pattern.sub("link{}".format(counter + i), string, count=1)
    counter += num_matches
    return string, counter



def lemmatize(df, col, dir):
    lemmatizer = WordNetLemmatizer()
    lemmatized_posts = list()
    for i, row in df.iterrows():
        text = row[col].split()
        tags = nltk.pos_tag(text)
        words = list()
        for i in range(len(tags)):
            if len(get_wordnet_pos(tags[i][1])) == 1:
                words.append(lemmatizer.lemmatize(text[i], get_wordnet_pos(tags[i][1])))
            else:
                words.append(lemmatizer.lemmatize(text[i]))
        new_text = " ".join(words)
        lemmatized_posts.append(new_text)
    df[col] = pd.Series(lemmatized_posts, index=df.index)
    return df


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

def english(df, col, data_dir):
    print("Cleaning text (transliterate all unicode to ascii-approx) and removing non-English posts")
    delete_indices = list()  # for non-English documents
    c = 0

    for i, row in df.iterrows():
        c += 1
        if c % 1000 == 0:
            print("Processed {} rows".format(c))
            print("{0:.3f}% done".format(100 * float(c) / df.shape[0]))
        text = row[col]
        decoded_text = unidecode(text).lower()
        collapsed_whitespace = re.sub(r"[\s]+", " ", decoded_text)
        if len(collapsed_whitespace) < 3:
            delete_indices.append(i)  # text is too short
            continue
        detected_language = TextBlob(collapsed_whitespace).detect_language()
        if detected_language != 'en':
            delete_indices.append(i)
        df.at[i, col] = collapsed_whitespace

    old_rows = df.shape[0]
    df = df.drop(delete_indices)
    print("Dropped {} rows; new dataframe has {} rows".format(old_rows - df.shape[0], df.shape[0]))
    return df

def stop_words(df, col, data_dir):
    stop_words = set(stopwords.words('english'))
    for i, row in df.iterrows():
        df.at[i, col] = " ".join(word for word in word_tokenize(row[col]) if not word in stop_words)
    return df
