import sys

sys.path.append('.')

from ntap.data import Dataset
from ntap.models import RNN


def initialize_dataset():
    data = Dataset("/var/lib/jenkins/workspace/ntap_data/data_alm.pkl")
    data.clean("text")
    data.set_params(vocab_size=5000, mallet_path = "/var/lib/jenkins/workspace/ntap_data/mallet-2.0.8/bin/mallet", glove_path = "/var/lib/jenkins/workspace/ntap_data/glove.6B/glove.6B.300d.txt")
    return data

def initialise_rnn(data, test_case_no):
    if test_case_no==1:
        model = RNN("authority ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==2:
        model = RNN("authority ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==3:
        model = RNN("authority ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==4:
        model = RNN("authority ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==5:
        model = RNN("authority ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==6 or test_case_no==14:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==7:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='adagrad', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==8:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='rmsprop', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==9:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==10:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ seq(text)", data=data, optimizer='sgd', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==11:
        model = RNN("authority ~ seq(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==12:
        model = RNN("authority ~ tfidf(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==13:
        model = RNN("authority ~ lda(text)", data=data, optimizer='adam', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==15:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ tfidf(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    elif test_case_no==16:
        model = RNN("authority+care+fairness+loyalty+purity+moral ~ lda(text)", data=data, optimizer='momentum', learning_rate=0.01, rnn_pooling=100)
    return model



def rnn_train(model):
    model.train(data, num_epochs = 5, model_path=".")


if __name__== '__main__':

    no_test_cases = 16
    for test_case_no in range(1,no_test_cases+1):
        try:
            data = initialize_dataset()
            print("\nChecking use-case: "+str(test_case_no)+"\n")
            model = initialise_rnn(data, test_case_no)
            rnn_train(model)
            print("use-case sucessfully executed")
        except Exception as e:
            print("use-case unsucessful")
            print(e)
