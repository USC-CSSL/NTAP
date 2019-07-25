import sys

sys.path.append('../')

from ntap.data import Dataset
from ntap.models import RNN


def initialize_dataset():
    data = Dataset("/home/anirudh/cssl/data/alm_data.pkl")
    data.clean("text")
    data.set_params(vocab_size=5000, mallet_path = "/home/anirudh/cssl/data/mallet-2.0.8/bin/mallet", glove_path = "/home/anirudh/cssl/data/glove.6B/glove.6B.300d.txt")
    return data

def initialise_rnn_1(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_2(data):
    model = RNN("authority+care+fairness+loyalty ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_3(data):
    model = RNN("care+fairness ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_4(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ tfidf(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_5(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ lda(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_6(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    return model


def initialise_rnn_7(data):
    model = RNN("loyalty ~ tfidf(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_8(data):
    model = RNN("purity ~ lda(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    return model

def rnn_train(model):
    model.train(data, num_epochs = 5, model_path=".")


if __name__== '__main__':

    try:
        data = initialize_dataset()
        print("\nChecking test-case 1\n")
        model = initialise_rnn_1(data)
        rnn_train(model)
        print("Test Case ran sucessfully")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 2\n")
        model = initialise_rnn_2(data)
        rnn_train(model)
        print("Test Case ran sucessfully")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 3\n")
        model = initialise_rnn_3(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as the implementation of MomentumOptimizer for RNN is missing an argument 'momentum'
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 4\n")
        model = initialise_rnn_4(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as TF-IDF for RNN is not implemented
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 5\n")
        model = initialise_rnn_5(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as LDA for RNN is not implemented
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 6\n")
        model = initialise_rnn_6(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as SGD for RNN is not implemented
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 7\n")
        model = initialise_rnn_7(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as TF-IDF for RNN is not implemented
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking test-case 8\n")
        model = initialise_rnn_8(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as LDA for RNN is not implemented
        print(e)
