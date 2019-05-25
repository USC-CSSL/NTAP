import operator
from nltk import tokenize as nltk_token
import pandas as pd
import os, math
import numpy as np

def get_batches(batch_size, vocab, corpus_ids, labels=None, features=None, padding=True):
    batches = []
    for idx in range(len(corpus_ids) // batch_size + 1):
        labels_batch = labels[idx * batch_size: min((idx + 1) * batch_size,
                            len(labels))] if labels is not None else []

        text_batch = corpus_ids[idx * batch_size: min((idx + 1) * batch_size,
                            len(corpus_ids))]

        features_batch = features[idx * batch_size: min((idx + 1) * batch_size,
                            len(features))] if features is not None else []

        lengths = np.array([len(line) for line in text_batch])
        if padding:
            text_batch = performPadding(vocab, text_batch)
        if len(text_batch) > 0:
            batches.append({"text": np.array([np.array(line) for line in text_batch]),
                            "sent_lens": lengths,
                            "label": np.array(labels_batch),
                            "feature": np.array(features_batch)})
    return batches

def graph(vectors, labels):
    pca = PCA(n_components=2)
    vec_components = pca.fit_transform(vectors)
    df = pd.DataFrame(data=vec_components, columns=['component 1', 'component 2'])
    finalDf = pd.concat([df, labels], axis=1)
    return finalDf

def loadEmbeddings(word_embedding, vocab, glove_path, embeddings_path, embedding_size):
    if word_embedding == 'glove':
        embeddings = loadGlove(vocab, glove_path, embeddings_path, embedding_size)
    return embeddings

def learnVocab(corpus, vocab_size):
    print("Learning vocabulary of size %d" % (vocab_size))
    tokens = dict()
    for sent in corpus:
        for token in sent:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    vocab = list(words[:vocab_size]) + ["<unk>", "<pad>"]
    return vocab

def loadGlove(vocab, glove_path, embeddings_path, embedding_size):
    if not os.path.isfile(glove_path):
        raise IOError("You're trying to access an embeddings file that doesn't exist")
    embeddings = dict()
    with open(embeddings_path, 'r', encoding='utf-8') as fo:
        glove = dict()
        for line in fo:
            tokens = line.split()
            embedding = np.array(tokens[len(tokens) - embedding_size:], dtype=np.float32)
            token = "".join(tokens[:len(tokens) - embedding_size])
            glove[token] = embedding
    unk_embedding = np.random.rand(embedding_size) * 2. - 1.
    if vocab is None:
        print("Error: Build vocab before loading GloVe vectors")
        exit(1)
    not_found = 0
    for token in vocab:
        try:
            embeddings[token] = glove[token]
        except KeyError:
            not_found += 1
            embeddings[token] = unk_embedding
    print(" %d tokens not found in GloVe embeddings" % (not_found))
    embeddings = np.array(list(embeddings.values()))
    return embeddings



def performPadding(vocab, corpus):
    padd_idx = vocab.index("<pad>")
    for i in range(len(corpus)):
        while len(corpus[i]) < max(len(sent) for sent in corpus):
            corpus[i].append(padd_idx)
    return corpus

def tokenize_data(corpus, text_col, max_length, min_length):
    drop = list()
    for i, row in corpus.iterrows():
        tokens = nltk_token.WordPunctTokenizer().tokenize(row[text_col].lower())
        corpus.at[i, text_col] = tokens[:min(max_length, len(tokens))]
        if len(row[text_col].split()) < min_length:
            drop.append(i)
    return corpus


def tokens_to_ids(corpus, vocab, learn_max=True):
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
