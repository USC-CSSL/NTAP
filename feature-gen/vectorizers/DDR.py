

from utils import load_glove_from_file

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from gensim.models import KeyedVectors as kv
class DDRVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, training_corpus, embedding_type,
                 tokenizer, dictionary, data_path, similarity):
        self.corpus = training_corpus
        self.embedding_type = embedding_type
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.corpus_path = os.path.join(data_path, 'word_embeddings') 
        dictionary_path = os.path.join(data_path, 'dictionaries', dictionary + '.json')
        try:
            with open(dictionary_path, 'r') as fo:
                data = json.load(fo)
                self.dictionary = data.items()
        except FileNotFoundError:
            print("Could not load dictionary %s from %s" % (self.dictionary, dictionary_path))
            exit(1)

        self.similarity = similarity
        self.corpus_filenames = {'GoogleNews': 'GoogleNews-vectors-negative300.bin',
                                 'common_crawl': 'glove.42B.300d.txt',
                                 'wiki_gigaword': 'glove.6B.300d.txt'}

    def get_feature_names(self):
        return [item[0] for item in self.dictionary]

    def get_document_avg(self, tokens, embed_size=300, min_threshold=0):
        arrays = list()
        oov = list()
        count = 0
        for token in tokens:
            if self.embedding_type == 'skipgram':
                try:
                    array = self.skipgram_vectors.get_vector(token)
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            elif self.embedding_type == 'GloVe':
                try:
                    array = self.glove_vectors[token]
                    arrays.append(array)
                    count += 1
                except KeyError as error:
                    oov.append(token)
            else:
                raise ValueError("Incorrect embedding_type specified; only possibilities are 'skipgram and 'GloVe'")
        if count <= min_threshold:
            return np.zeros(embed_size), oov
        sentence = np.array(arrays)
        mean = np.mean(sentence, axis=0)
        return mean, oov

    def fit(self, X, y=None):
        """
        Load selected word embeddings based on specified name 
            (raise exception if not found)
        """

        self.embeddings_path = os.path.join(self.corpus_path, self.embedding_type, 
                                            self.corpus_filenames[self.corpus])
        if not os.path.exists(self.embeddings_path):
            print(self.embeddings_path)
            raise ValueError("Invalid Word2Vec training corpus + method")
        print("Loading embeddings")
        if self.embedding_type == 'skipgram':
            # type is 'gensim.models.keyedvectors.Word2VecKeyedVectors'
            self.skipgram_vectors = kv.load_word2vec_format(self.embeddings_path, binary=True)
        if self.embedding_type == 'GloVe':
            # type is dict
            self.glove_vectors = load_glove_from_file(self.embeddings_path)
        return self

    def transform(self, raw_docs, y=None):
        print("Calculating dictionary centers")
        concepts = list()
        for concept, words in self.dictionary:
            concept_mean, _ = self.get_document_avg(words)
            concepts.append(concept_mean)
        ddr_vectors = list()
        for sentence in raw_docs:
            tokens = self.tokenizer(sentence)
            sentence_mean, oov = self.get_document_avg(tokens)    
            outputs = [self.similarity(sentence_mean, concept) for concept in concepts]
            ddr_vectors.append(outputs)
        X = np.array(ddr_vectors)
        X = np.nan_to_num(X)
        return X
