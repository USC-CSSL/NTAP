
import logging, os, operator
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models.wrappers import LdaMallet  
from gensim import corpora
from gensim.utils import simple_preprocess  # removes short words, converts to lowercase

# mallet_path = os.environ["MALLET"]

class LDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, save_dir, tokenizer, mallet_path, num_topics=100, num_iter=100, seed=0,
                 num_words=10000, stop_words='english', ngram=[0, 1]):
        self.save_dir = save_dir  # to log training and to save trained model
        self.save_dir = os.path.join(save_dir, "lda/")  # temporary
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.num_topics = num_topics
        self.num_iter = num_iter
        self.random_seed = seed
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.stop_words = stop_words
        self.mallet_path = mallet_path
    
    def fit(self, X, y=None):
        docs = [simple_preprocess(doc) for doc in X]
        self.dictionary = corpora.Dictionary(docs)
        return self  ## DEBUG -- LEIGH 2/5/19 (with Brendan)

    def transform(self, X, y=None):
        #TODO: Revise with user-specified tokenization
        docs = [self.dictionary.doc2bow(simple_preprocess(doc)) for doc in X]
        model = LdaMallet(mallet_path=self.mallet_path,
                          id2word=self.dictionary, 
                          prefix=self.save_dir,
                          num_topics=self.num_topics, 
                          iterations=self.num_iter,
                          optimize_interval=20)
        model.train(docs)
        doc_topics = list()
        for doc_vec in model.read_doctopics(model.fdoctopics()):
            topic_ids, vecs = zip(*doc_vec)
            doc_topics.append(np.array(vecs))
        return np.array(doc_topics)

