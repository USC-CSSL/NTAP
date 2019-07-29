import sys

sys.path.append('.')

from ntap.data import Dataset
from ntap.models import RNN


def initialize_dataset():
    data = Dataset("/var/lib/jenkins/workspace/ntap_data/data_alm.pkl")
    data.clean("text")
    data.set_params(vocab_size=5000, mallet_path = "/var/lib/jenkins/workspace/ntap_data/mallet-2.0.8/bin/mallet", glove_path = "/var/lib/jenkins/workspace/ntap_data/glove.6B/glove.6B.300d.txt")
    return data

def initialise_rnn_1(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_2(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_3(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_4(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_5(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_6(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_7(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_8(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_9(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_10(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_11(data):
    model = RNN("authority ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_12(data):
    model = RNN("authority ~ tfidf(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_13(data):
    model = RNN("authority ~ lda(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_15(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ tfidf(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model

def initialise_rnn_16(data):
    model = RNN("authority+care+fairness+loyalty+purity+moral ~ lda(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model



def rnn_train(model):
    model.train(data, num_epochs = 5, model_path=".")


if __name__== '__main__':

    try:
        data = initialize_dataset()
        print("\nChecking use-case 1\n")
        model = initialise_rnn_1(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 2\n")
        model = initialise_rnn_2(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 3\n")
        model = initialise_rnn_3(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 4\n")
        model = initialise_rnn_4(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as the implementation of MomentumOptimizer for RNN is missing an argument 'momentum'
        print("use-case unsucessful")
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 5\n")
        model = initialise_rnn_5(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as implementation of SGDOptimizer for RNN is not implemented
        print("use-case unsucessful")
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 6 and 14\n")
        model = initialise_rnn_6(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 7\n")
        model = initialise_rnn_7(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 8\n")
        model = initialise_rnn_8(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)


    try:
        data = initialize_dataset()
        print("\nChecking use-case 9\n")
        model = initialise_rnn_9(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as the implementation of MomentumOptimizer for RNN is missing an argument 'momentum'
        print("use-case unsucessful")
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 10\n")
        model = initialise_rnn_10(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as implementation of SGDOptimizer for RNN is not implemented
        print("use-case unsucessful")
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 11\n")
        model = initialise_rnn_11(data)
        rnn_train(model)
        print("use-case sucessfully executed")
    except Exception as e:
        print(e)


    try:
        data = initialize_dataset()
        print("\nChecking use-case 12\n")
        model = initialise_rnn_12(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as TF-IDF for RNN is not implemented
        print("use-case unsucessful")
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 13\n")
        model = initialise_rnn_3(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as LDA for RNN is not implemented
        print("use-case unsucessful")
        print(e)


    try:
        data = initialize_dataset()
        print("\nChecking use-case 15\n")
        model = initialise_rnn_15(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as TF-IDF for RNN is not implemented
        print("use-case unsucessful")
        print(e)

    try:
        data = initialize_dataset()
        print("\nChecking use-case 16\n")
        model = initialise_rnn_16(data)
        rnn_train(model)
    except Exception as e:
        # Test case results in error as as LDA for RNN is not implemented
        print("use-case unsucessful")
        print(e)
