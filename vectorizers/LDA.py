
#import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

class LDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, seed, tokenizer, num_topics=100, 
                    num_iter=100, num_words=10000, stop_words='english', ngram=[0, 1]):
        self.num_topics = num_topics
        self.num_iter = num_iter
        self.random_seed = seed
        self.tokenizer = tokenizer
        self.num_words = num_words
        self.stop_words = stop_words
        self.ngram = ngram
    
    def fit(self, X, y=None):
        # find best model by cross-validation (grid-search params)
        self.dt_matrix = CountVectorizer(tokenizer=self.tokenizer,
                                         stop_words=self.stop_words,
                                         max_features=self.num_words, ngram_range=self.ngram).fit_transform(X)

        lda_model = LatentDirichletAllocation(n_components=self.num_topics, learning_method='online',
                                             max_iter=self.num_iter, verbose=1,
                                             random_state=self.random_seed)
        print("Fitting LDA model with 3-fold cross-validation")
        print("Tuning: learning_decay, learning_offset, batch_size")
        """
        choose_lda = GridSearchCV(lda_model, cv=3, iid=True,
                                  param_grid={"learning_decay": np.arange(0.7, 0.9, 0.05),
                                              "learning_offset": np.arange(10, 50, 20),
                                              "batch_size": [32,64,128,256]
                                             })
        """
        self.lda = lda_model.fit(self.dt_matrix)
        print(self.lda.components_)
        return self

    def transform(self, X, y=None):
        return self.lda.transform(self.dt_matrix)

    def get_feature_names(self):
        return ["topic" + str(i) for i in range(self.num_topics)]
