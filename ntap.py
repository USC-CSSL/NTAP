
import pandas as pd
import json
from nltk import tokenize as nltk_token

link_re = re.compile(r"(http(s)?[^\s]*)|(pic\.[s]*)")
hashtag_re = re.compile(r"#[a-zA-Z0-9_]+")
mention_re = re.compile(r"@[a-zA-Z0-9_]+")

nltk_tokenizer = nltk_token.TreebankWordTokenizer()
treebank_tokenizer = nltk_token.TreebankWordTokenizer()
wordpunc_tokenizer = nltk_token.WordPunctTokenizer()
#from tokenization.happierfuntokenizing import HappierTokenizer
#happierfun not currently supported

def wordpunc_tokenize(text):
    return wordpunc_tokenizer.tokenize(text)
#def happiertokenize(text):
    #tok = HappierTokenizer(preserve_case=False)
    #return tok.tokenize(text)
def tweettokenize(text):
    # keeps #s and @s appended to their next word
    return nltk_tweettokenize.tokenize(text)

def read_file(path):
    ending = path.split('.')[-1]
    if ending == 'csv':
        return pd.read_csv(path)
    elif ending == 'tsv':
        return pd.read_csv(path, delimiter='\t')
    elif ending == 'pkl':
        return pd.read_pickle(path)
    elif ending == 'json':
        return pd.read_json(path)

class Dataset:
    def __init__(self, path, tokenizer='wordpunct', vocab_size=5000, embed='glove', min_token=5):
        try:
            self.data = read_file(path)
            print("Loaded file with {} documents".format(len(self.data)))
        except Exception as e:
            print("Could not read data from {}".format(path))
            print("Exception:", e)
            return
        self.learn_vocab()

    def clean(self, column, remove=["hashtags", "mentions", "links"], mode='remove'):
        if column not in self.data:
            raise ValueError("{} not in dataframe".format(column))
        def mentions(t):
            return mention_re.sub("", t)
        def links(t):
            return link_re.sub("", t)
        def hashtags(t):
            return hashtag_re.sub("", t)

        for pattern in pat_type:
            if pattern == "mentions":
                self.data[col] = self.data[col].apply(mentions)
            if pattern == "hashtags":
                self.data[col] = self.data[col].apply(hashtags)
            if pattern == "links":
                self.data[col] = self.data[col].apply(links)
        print("TODO: replace dataframe with removed rows")

    def clean_target(self, target, variable_type='factor', 
            null_val=None):
        print("TODO: remove NaNs/Nulls (store new data) and encode variable (one-hot, etc.)")
    
    def tokenize(self, column, stopwords=None, max_len=None, 
            vocab_size=None, lower=True, tokenizer=None):
        print("TODO: tokenize and save to self.words")
        if tokenizer is not None:
            self.tokenizer = toks[tokenizer] # TODO
        if vocab_size is not None:
            self.learn_vocab(vocab_size)
        tokenized, drop = list(), list()
        for idx, string in self.data[column].iteritems:  #TODO: check
            tokens = self.tokenizer(string, lower, stopwords, max_len)
            # TODO: max_len truncation/padding
            tokenized.append(tokens)
            if len(tokens) < self.min_len:
                drop.append(idx)
        prev = len(self.data)
        self.data.drop(rows=drop, inplace=True)
        print("Dropped {} docs due to not enough tokens".format(len(prev - len(self.data))))
        return 

        # move following code to constructor of method (parse_formula)
        X = np.array(tokens_to_ids(text, vocab))
        y = np.transpose(np.array([np.array(train_data[target].astype(int)) for target in getTargetColumnNames(all_params)]))

    def lda(self, column, stopwords=None, method='mallet', num_topics=20, max_iter=500, save_model=None, load_model=None):
        print("TODO: implement lda features\nSave to self.lda")

    def tfidf(self, column, stopwords=None, vocab_size, **kwargs):
        print("TODO: implement tfidf, save to self.tfidf")

    def ddr(self, column, dictionary, embed='glove', **kwargs):
        print("TODO: implement ddr features")

    def bert(self, some_params):
        print("TODO: implement a featurization based on BERT")


    def tokens_to_ids(corpus, vocab, learn_max=True):
        # TODO: incorporate into 'tokenize' (one-step)
        print("Converting corpus of size %d to word indices based on learned vocabulary" % len(corpus))
        if vocab is None:
            raise ValueError("learn_vocab before converting tokens")

        mapping = {word: idx for idx, word in enumerate(vocab)}
        unk_idx = vocab.index("<unk>")
        for i in range(len(corpus)):
            row = corpus[i]
            for j in range(len(row)):
                try:
                    corpus[i][j] = mapping[corpus[i][j]]
                except:
                    corpus[i][j] = unk_idx
        if learn_max:
            max_length = max([len(line) for line in corpus])
        return corpus

    def write(self, formatting='.json'):
        dest = os.path.join(self.dest, self.filename + formatting)
        if formatting == '.json':
            self.data.to_json(dest)
        if formatting == '.csv':
            self.data.to_csv(dest)
        if formatting == '.tsv':
            self.data.to_csv(dest, sep='\t')
        if formatting == '.pkl':
            self.data.to_pickle(dest)
        if formatting == '.stata':
            self.data.to_stata(dest)
        if formatting == '.hdf5':
            self.data.to_hdf(dest)
        if formatting == '.excel':
            self.data.to_excel()
        if formatting == '.sql':
            self.data.to_sql()
