
def load_glove_from_file(fname, vocab=None):
    # vocab is possibly a set of words in the raw text corpus
    if not os.path.isfile(fname):
        raise IOError("You're trying to access a GloVe embeddings file that doesn't exist")
    embeddings = dict()
    with open(fname, 'r') as fo:
        for line in fo:
            tokens = line.split()
            if vocab is not None:
                if tokens[0] not in vocab:
                    continue
            if len(tokens) > 0:
                embeddings[str(tokens[0])] = np.array(tokens[1:], dtype=np.float32)
    return embeddings
