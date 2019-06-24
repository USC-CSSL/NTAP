import operator
from nltk import tokenize as nltk_token
import pandas as pd
import os, math
import numpy as np

def checkHyperParameters(params):
    neural_params = getNeuralParams(params)
    hyperParams = getHyperParameters(params)
    for param in hyperParams:
        if len(neural_params[param])>1:
            return True
    return False
# a function to get the name of Base Directory and the input file name
def getBaseDirAndFilename(params):
    base_dir, filename = os.path.split(getInputFilePath(params))
    return base_dir, filename

# a function to get the list of the baseline features
def getBaselineFeaturesList(params):
    return params['baseline']['features']

# a function to get the baseline method
def getBaselineMethod(params):
    return params['baseline']['method']

# a function to get the baseline targets
def getBaselineTargets(params):
    return params['baseline']['targets']

# a function to get the feature filename
def getFeatureFileName(params):
    return params['model']['feature']

# a function to get the number of Cross Validation folds specified
def getFolds(params):
    neural_params = getNeuralParams(params)
    return neural_params["kfolds"]

# a function to get the glove plath
def getGlovePath(params):
    return params['path']['glove_path']

def getHyperParameters(params):
    neural_params = getNeuralParams(params)
    #params = ["learning_rate","keep_ratio"]
    #params = ["learning_rate"]
    return neural_params["hyper_parameters"]

# a function to get the Input file path
def getInputFilePath(params):
    return params["processing"]["input_path"]

# a function to get the learning rate set
def getLearningRate(params):
    neural_params = getNeuralParams(params)
    return neural_params['learning_rate']

# a function to get the maximum sequence length
def getMaxLength(params):
    neural_params = getNeuralParams(params)
    return neural_params["max_length"]

# a function to get the minimum sequence length
def getMinLength(params):
    neural_params = getNeuralParams(params)
    return neural_params["min_length"]

# a function to get the name of the model specified
def getModel(params):
    neural_params = getNeuralParams(params)
    return neural_params["model"]

# a function to get the directory where best model is stored
def getModelDirectory(params):
    base_dir, filename = getBaseDirAndFilename(params)
    return base_dir+"/MFTC_models/"+filename.split(".")[0]+"/"+getModel(params)+"/NTAP_model"

# a function to get the name of the model method specified
def getModelMethod(params):
    return params['model']['method']

# a function to get the neural params
def getNeuralParams(params):
    return params['neural_params']

# a function to get the truth value of the "predict" parameter
def getPredictParam(params):
    neural_params = getNeuralParams(params)
    return neural_params["predict"]

# a function to get processing tasks in clean
def getPreProcessingCleanList(params):
    return params["processing"]["clean"]

# a function to get the list of preprocessing jobs
def getPreProcessingJobList(params):
    return params['processing']['jobs']

# a function to get the list of target names
def getTargetColumnNames(params):
    return params["neural_params"]["target_cols"]

# a function to get the truth value of the "train" parameter
def getTrainParam(params):
    neural_params = getNeuralParams(params)
    return neural_params["train"]

def getTextColName(params):
    return params['processing']['text_col']

def getTestFilePath(params):
    return params['model']['test_filepath']
# a function to get the vocab size specified
def getVocabSize(params):
    neural_params = getNeuralParams(params)
    return neural_params["vocab_size"]

# a function to get the word2vec path
def getWordtoVecPath(params):
    return params['path']['word2vec_path']

# a function to divide the dataset into batches
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

# a function to load the word embeddings
def loadEmbeddings(word_embedding, vocab, glove_path, embeddings_path, embedding_size):
    if word_embedding == 'glove':
        embeddings = loadGlove(vocab, glove_path, embeddings_path, embedding_size)
    return embeddings

# a function to build the vocabulary
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

# a function to load the Glove word embeddings
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

# a function that adds padding
def performPadding(vocab, corpus):
    padd_idx = vocab.index("<pad>")
    for i in range(len(corpus)):
        while len(corpus[i]) < max(len(sent) for sent in corpus):
            corpus[i].append(padd_idx)
    return corpus

# a function to set the feature size
def setFeatureSize(params, feat):
    neural_params = getNeuralParams(params)
    neural_params["feature_size"] = feat

# a function to get the task given (train or predict)
def getTask(params):
    neural_params = getNeuralParams(params)
    return neural_params["task"].lower()

# a function to tokenize the data
def tokenize_data(corpus, text_col, max_length, min_length):
    drop = list()
    for i, row in corpus.iterrows():
        tokens = nltk_token.WordPunctTokenizer().tokenize(row[text_col].lower())
        corpus.at[i, text_col] = tokens[:min(max_length, len(tokens))]
        if len(row[text_col].split()) < min_length:
            drop.append(i)
    return corpus


# a function that coverts corpus to word indices based on Vocab learnt
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
