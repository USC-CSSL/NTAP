from .bagofwords import Dictionary


class GloVe:
    def __init__(self, glove_path):
        pass


class DDR:
    def __init__(self, corpus, dic, glove_path, **kwargs):
        dictionary = Dictionary(dic)

        for word_list in name:
            for w in word_list:
                if w not in self.mapping:
                    self.mapping[w] = len(self.mapping)
                    self.vocab.append(w)

        embeddings, _ = self.__read_glove(self.glove_path)

        dictionary_centers = {cat: self.__embedding_of_doc(words, embeddings) \
                for cat, words in zip(dictionary, name)}
        features = list()
        for doc in self.data[column].values.tolist():
            e = self.__embedding_of_doc(doc, embeddings=embeddings)
            not_found = False
            if e.sum() == 0:  # no tokens found
                not_found = True
            doc_feat = dict()
            for category, dict_vec in dictionary_centers.items():
                if not_found:
                    e = np.random.rand(len(e))
                doc_feat[category] = cosine(dict_vec, e)
            features.append(doc_feat)

        features = pd.DataFrame(features)
        features, categories = features.values, list(features.columns)
        self.features["ddr"] = np.array(features)  # type numpy array
        self.feature_names["ddr"] = categories # list of strings

    def fit(self, X, y=None):
        pass
    def transform(self, X):
        pass
    """
    def __embedding_of_doc(self, doc_string, embeddings, agg='mean', thresh=1):
        tokens = self.__tokenize_doc(doc_string)
        embedded = list()
        for t in tokens:
            if t in self.mapping:
                embedded.append(embeddings[self.mapping[t]])
        if len(embedded) < thresh:
            return np.zeros(embeddings.shape[1])
        return np.array(embedded).mean(axis=0)
    """
