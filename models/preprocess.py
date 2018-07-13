import re, os, json, nltk, string, emot, emoji
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize
from unidecode import unidecode
from textblob import TextBlob

# alpha_re = re.compile(r"[^a-zA-Z\s]")
# length_re = re.compile(r'\w{3,}')

def preprocess_text(df,
                    col,
                    methods,
                    data_dir ):

    priority = {1: ["link", "hashtag", "emojis", "mentions"],
                2: ["all_alpha", "lemmatize", "stem", "stop_words", "partofspeech", "ascii"]
                }

    pre_list = [item for sublist in [val for val in priority.values()] for item in sublist]

    if not set(methods).issubset(pre_list):
        print("Some preprocessing methods are not available")
        exit(1)

    if col not in df.columns:
        print("{} is not a column name".format(col))
        exit(1)

    for pri in priority.keys():
        for method in methods:
            if method in priority[pri]:
                df = globals()[method](df, col, data_dir)

    df = remove_whitespaces(df, col)
    return df



def remove_whitespaces(df, col):
    for i, row in df.iterrows():
        text = row[col]
        collapsed_whitespace = re.sub(r"[\s]+", " ", text)
        df.at[i, col] = collapsed_whitespace
    return df


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

def hashtag(df, col, data_dir):
    print("Extracting hashtags")
    hashtags = list()
    for index, row in df.iterrows():
        text = row[col]
        post_hashtags = [word[1:] for word in text.split() if word[0] == "#"]
        hashtags.append(post_hashtags)
    df['hashtags'] = pd.Series(hashtags, index=df.index)
    return df


def sub_link(string, links, counter=0):
    http_pattern = re.compile(r"http(s)?[^\s]+")
    num_matches = len(http_pattern.findall(string))
    for i in range(num_matches):
        links[counter + i] = http_pattern.search(string).group(0)
        string = http_pattern.sub("link{}".format(counter + i), string, count=1)
    counter += num_matches
    return string, counter

def emojis(df, col, dir):
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

def partofspeech(df, col, dir):
    useless_parts = []
    for i, row in df.iterrows():
        tags = nltk.pos_tag(row[col].split())
        new_text = list()
        for tag in tags:
            if tag[1] not in useless_parts:
                new_text.append(tag[0])
        df.at[i, col] = " ".join(word for word in new_text)
    return df

def stem(df, col, dir):
    stemmer = PorterStemmer()
    for i, row in df.iterrows():
        new_text = " ".join([stemmer.stem(w) for w in row[col].split()])
        df.at[i, col] = new_text
    return df

def lemmatize(df, col, dir):
    lemmatizer = WordNetLemmatizer()
    #lemmatized_posts = list()
    for ind, row in df.iterrows():
        text = word_tokenize(row[col])
        tags = nltk.pos_tag(text)
        words = list()
        for i in range(len(tags)):
            if len(get_wordnet_pos(tags[i][1])) == 1:
                words.append(lemmatizer.lemmatize(text[i], get_wordnet_pos(tags[i][1])))
            else:
                words.append(lemmatizer.lemmatize(text[i]))
        new_text = " ".join(words)
        #lemmatized_posts.append(new_text)
        df.at[ind, col] = new_text
    #df[col] = pd.Series(lemmatized_posts, index=df.index)
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

def ascii(df, col, data_dir):
    print("Transliterate all unicode to ascii-approx")
    for i, row in df.iterrows():
        text = row[col]
        decoded_text = unidecode(text).lower()
        collapsed_whitespace = re.sub(r"[\s]+", " ", decoded_text)
        df.at[i, col] = collapsed_whitespace
    return df


def stop_words(df, col, data_dir):
    stop_words = set(stopwords.words('english'))
    for i, row in df.iterrows():
        df.at[i, col] = " ".join(word for word in word_tokenize(row[col]) if not word in stop_words)
    return df


def mentions(df, col, data_dir):
    for i, row in df.iterrows():
        df.at[i, col] = " ".join(word for word in word_tokenize(row[col]) if word[0] != "@")
    return df


def all_alpha(df, col, data_dir):
    r = re.compile('[^a-zA-Z]')
    for i, row in df.iterrows():
        new_text = " ".join(r.sub("", word) for word in word_tokenize(row[col]))
        df.at[i, col] = new_text
    return df
