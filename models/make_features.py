import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, CategoricalEncoder
from sklearn_pandas import gen_features, CategoricalImputer
from sklearn.impute import SimpleImputer as Imputer

from scipy import spatial

from vectorizers import BoMVectorizer, DDRVectorizer, LDAVectorizer
from nltk import tokenize as nltk_token
nltk_tokenizer = nltk_token.TreebankWordTokenizer()
alpha_re = re.compile(r"[^a-zA-Z\s]")
length_re = re.compile(r'\w{3,}')


def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)

def tokenize(text):
    tokens = nltk_tokenizer.tokenize(text)
    return tokens

def prep_text(s):
    text = " ".join(length_re.findall(alpha_re.sub('', s))).lower()
    return text

def add_categorical(transformer_list, ordinal_cols, categorical_cols):

    if len(categorical_cols) > 0: 
            transformer_list += gen_features(
                    columns=[ [col] for col in categorical_cols],
                    classes=[CategoricalImputer, CategoricalEncoder]
                            )
    if len(ordinal_cols) > 0:
    	transformer_list += gen_features(
                    columns=[ [col] for col in ordinal_cols],
                    classes=[{'class': Imputer, 'missing_values':-1},
                             {'class': MinMaxScaler}
                            ])
    return transformer_list 

# returns transformer list, one per generated/loaded text feature
def get_text_transformer(dataframe,
                         data_dir,
                         text_col,  # can be list (if loading features) or string (if generating)
                         method,  # type of features to load/generate
                         punc=False,  # remove punctuation in preprocessing
                         clean_type='alpha',  # valid characters for a 'word'
                         bom_method=None,  # options: 'skipgram', 'glove'
                         training_corpus=None,  # options: 'google-news', 'wiki', 'common-crawl'
                         dictionary=None,  # options: 'liwc', 'mfd'
                         comp_measure=None, # options: cosine-sim
                         random_seed=-1
                         ):
    # Either generates features from text (tfidf, skipgram, etc.) or load from file

    already_features = type(text_col) == list
    gen_list = ['tfidf', 'bag-of-means', 'ddr', 'lda', 'fasttext', 'infersent']

    if not already_features:
        if method not in gen_list:
            print("If generating features from text, specify \'text_col\' as a string object")
            exit(1)
        if text_col not in dataframe.columns:
            print("Not valid text column")
            exit(1)
    if already_features:
        if method != 'load':
            print("If loading features from text, specify \'text_col\' as a list (of columns)")
            exit(1)
        if not set(text_col).issubset(dataframe.columns):
            print("To load LIWC/MFD features, load dataframe with \'text_col\' as columns")
            exit(1)

    transformers = list()
    if method == 'tfidf':
        # process text
        transformers.append((text_col, TfidfVectorizer(min_df=10, 
                                                   stop_words='english',
                                                   preprocessor=prep_text,
                                                   tokenizer=tokenize), {'alias': 'tfidf'}
                            ))
    elif method == 'load_features':
        if len(text_col) > 0:
            transformer_list += gen_features(
                    columns=[ [col] for col in text_col],
                    classes=[StandardScaler])

    elif method == 'lda':
        num_topics = 100
        transformers.append((text_col, LDAVectorizer(seed=random_seed,
                                                     tokenizer=tokenize,
                                                     preprocessor=prep_text,
                                                     num_topics=num_topics),
                             {'alias': method + "_" + str(num_topics) + "topics"}))
    elif method == 'bag-of-means':
        if training_corpus is None or bom_method is None:
            print("Specify bom_method and training_corpus")
            exit(1)
        transformers.append((text_col, BoMVectorizer(training_corpus,
                                                     embedding_type=bom_method,
                                                     remove_stopwords=True,
                                                     preprocessor=prep_text,
                                                     tokenizer=tokenize, data_path=data_dir)
                                     , {'alias': "_".join([method, bom_method])}
                                                     ))
    elif method == 'ddr':
        if dictionary is None or training_corpus is None or bom_method is None:
            print("Specify dictionary, bom_method, and training_corpus")
            exit(1)
        sim = cosine_similarity if comp_measure == 'cosine-sim' else None
        transformers.append((text_col, [DDRVectorizer(training_corpus,
                                                      embedding_type=bom_method,
                                                      remove_stopwords=True,
                                                      preprocessor=prep_text,
                                                      tokenizer=tokenize, 
                                                      data_path=data_dir,
                                                      dictionary=dictionary,
                                                      similarity=sim)
                                                      ], {'alias': "_".join([method, dictionary])}))
        # chain skipgram-bom with liwc transformer
    elif method == 'fasttext':  # exceeds at syntactic tasks
        print("FastText")
        # call fasttext executable with subprocess module
    elif method == 'infersent':
        transformers.append((feat, InferSentVectorizer(args)))
    else:
        print("Invalid method || exiting")
        exit(1)
    
    return transformers
