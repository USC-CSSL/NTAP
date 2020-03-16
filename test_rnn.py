import sys

sys.path.append('.')

from ntap.data import Dataset
from ntap.RNN import RNN
import pandas as pd
import argparse

def initialize_dataset(data_dir):
    data = Dataset(data_dir)
    data.set_params(vocab_size=10000,
                    glove_path = "glove.6B.300d.txt")
    data.clean("text")
    return data

def train_model(model, data, task):
    result = model.CV(data, num_epochs=10, num_folds=10)
    result.summary()
    #model.train(data, num_epochs=10, model_path="save/" + task)

def initialize_model(data, task):
    if task == "hate":
        model = RNN("hate ~ seq(text)",
                rnn_dropout=0.2, hidden_size=100, cell="biGRU",
                embedding_source="glove", data=data, optimizer='adam',
                learning_rate=0.0001)
    elif task == "bind":
        model = RNN("individualizing + binding ~ seq(text)",
                rnn_dropout=0.5, hidden_size=128, cell="biGRU",
                embedding_source="glove", data=data, optimizer='adam',
                learning_rate=0.001)
    else:
        model = RNN("care + harm + fairness + cheating + authority + subversion +"
            "loyalty + betrayal + purity + degradation ~ seq(text)",
            rnn_dropout=0.5, hidden_size=256, cell="biGRU",
            embedding_source="glove", data=data, optimizer='adam',
            learning_rate=0.001)
    return model

def predict(model, new_data, data, task):
    predictions = model.predict(new_data=new_data, orig_data=data,
                                model_path="save/" + task + "/" + task,
                                column="text")
    pd.DataFrame.from_dict(predictions).to_csv(task + "_predictions.csv")

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--task")
    args = parser.parse_args()

    data = initialize_dataset(args.data)
    model = initialize_model(data, args.task)
    train_model(model, data, args.task)


