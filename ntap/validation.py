import numpy as np
import pandas as pd

class _CheckParamsMixin:

    def _verify_corpus(self, corpus):
        assert isinstance(corpus, (list, pd.Series, np.ndarray)), "corpus must be list-like"

        if isinstance(corpus, (list, np.ndarray)):
            has_strings = len(corpus) == sum([isinstance(a, str)
                                              for a in corpus])
            assert has_strings, "corpus must contain type `str`"
        elif isinstance(corpus, pd.Series):
            has_strings = len(corpus) == sum([isinstance(a, str)
                                              for a in corpus.values])
            assert has_strings, "corpus must contain type `str`"

    def _verify_params(self, **kwargs):

        if 'vocab_size' in kwargs:
            assert isinstance(vocab_size, int), "vocab_size must be an integer"
        if 'max_df' in kwargs:
            assert max_df >= 0. and max_df <= 1, "max_df must be between 0 and 1"
        if 'min_df' in kwargs:
            assert isinstance(min_df, int) and min_df >= 0, "min_df must be int greater than 0"
        if 'tokenizer' in kwargs:
            assert True, "No op"
            #tokenizer_options = ", ".join(_TOKENIZERS.keys())
            #assert tokenizer in _TOKENIZERS, "valid tokenizers: {}".format(tokenizer_options)

    def _check_preprocess(self):
        if self.preprocessor:
            assert self.preprocessor, "No OP"

