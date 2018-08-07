import re, os, json, nltk, string, emot, emoji, sys
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize
from unidecode import unidecode
from textblob import TextBlob
from sys import stdout
import copy

# alpha_re = re.compile(r"[^a-zA-Z\s]")
# length_re = re.compile(r'\w{3,}')



def preprocess_text(df,
                    col,
                    methods,
                    config_text,
                    name = None
                    ):
    """
    priority = {1: ["link", "hashtag", "emojis", "mentions"],
                2: ["all_alpha", "lemmatize", "stem", "stop_words", "partofspeech", "ascii"]
                }
    
    pre_list = [item for sublist in [val for val in priority.values()] for item in sublist]

    print("restructuring... (Aida: I commented this part!)")

    if not set(methods).issubset(pre_list):
        print("Some preprocessing methods are not available")
        exit(1)

    if col not in df.columns:
        print("{} is not a column name".format(col))
        return df
        #exit(1)
    """

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

"""

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

"""

def stem(df, col):
    print("Stemming with Porter Stemmer")
    stemmer = PorterStemmer()
    texts = df[col].values.tolist()
    new_texts = [" ".join([stemmer.stem(w) for w in text.split()]) for text in texts]
    df.loc[:, col] = new_texts
    return df

"""

def lemmatize(df, col):
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

def ascii(df, col):
    print("Transliterate all unicode to ascii-approx")
    for i, row in df.iterrows():
        text = row[col]
        decoded_text = unidecode(text).lower()
        collapsed_whitespace = re.sub(r"[\s]+", " ", decoded_text)
        df.at[i, col] = collapsed_whitespace
    return df

"""

def stop_words(df, col, lower):
    print("Removing stop words")
    stop_words = set(stopwords.words('english'))
    stop_words_exp = re.compile(r"({})\s+".format('|'.join(stop_words)))
    texts = df[col].values.tolist()
    if lower:
        new_texts = [stop_words_exp.sub(' ', text.lower()) for text in texts]
    else:
        new_texts = [stop_words_exp.sub(' ', text) for text in texts]
    df.loc[:, col] = new_texts
    return df
    

if __name__ == '__main__':
    filename = os.environ['SOURCE_PATH']
    param_file = os.environ['PARAMS']
    dataframe = pd.read_pickle(filename)
    
    with open(param_file, 'r') as fo:
        params = json.load(fo)
    for par in params.keys():
        locals()[par] = params[par]
    
    for method in extract:
        globals()[method](dataframe, text_col)    
    for method in preprocess:
        globals()[method](dataframe, text_col)
    if stopwords is not None:
        stop_words(dataframe, text_col, lower)
    dataframe.to_pickle(filename)
