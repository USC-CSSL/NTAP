import pandas as pd
import os, json
import numpy as np

from neural.neural import Neural
from random import randint
from sklearn.model_selection import KFold
import tensorflow as tf

"""
if len(sys.argv) != 2:
    print("Usage: python run_neural.py params.json")
    exit(1)
with open(sys.argv[1], 'r') as fo:
    params = json.load(fo)

params = json.load(open("params/test_lstm_hate.json", "r"))
try:
    for par in params.keys():
        locals()[par] = params[par]


except KeyError:
    print("Could not load all parameters; if you're not using a parameter, set it to None")
    exit(1)



#Preprocessing the data


############### Loading non annotated data##############
print("Loading the whole dataframe")
init_clock = time.clock()

dataframe = pd.read_pickle(data_dir + '/' + saved_data) if saved else pd.read_pickle(data_dir + '/' + all_data)
print("Whole data has {} rows and {} columns".format(dataframe.shape[0], dataframe.shape[1]))

# Preprocessing the data
if not saved:
    dataframe = preprocess_text(dataframe, "text", preprocessing, data_dir, config_text, "cleaned_data")
print("Loading data took %d seconds " % (time.clock() - init_clock))

init_clock = time.clock()
docs = [tokenize(sent.lower()) for sent in dataframe["text"].values.tolist()]
print("Tokenizing data took %d seconds " % (time.clock() - init_clock))
######## Loading annotated data ############
print("Loading annotated dataframe")
annotated_df = pd.read_pickle(data_dir + '/' + saved_train) if saved else pd.read_pickle(data_dir + '/' + dataframe_name)

print("Annotated dataframe has {} rows and {} columns".format(annotated_df.shape[0], annotated_df.shape[1]))
init_clock = time.clock()
if not saved:
    annotated_df = preprocess_text(annotated_df, text_col, preprocessing, data_dir, config_text, "cleaned_train")
#annotated_df = annotated_df.sample(frac=1)
print("Loading annotated data took %d seconds " % (time.clock() - init_clock))


annotated_df = annotated_df.sample(frac = 1)
print("Number of data points in the sampled data is ", len(annotated_df))
"""

param_path = os.environ['PARAMS']
source_path = os.environ['SOURCE_PATH']
prediction_path = os.environ['PRED_PATH']

if __name__ == '__main__':
    with open(param_path, 'r') as fo:
        params = json.load(fo)
    source_df = pd.read_pickle(source_path)
    print(source_df.shape)
    missing_indices = list()
    for target in params["target_cols"]:
        print("Predicting {}".format(target))
        missing_indices.extend(source_df[source_df[target] == -1.].index)
    source_df = source_df.drop(missing_indices)
    print("Shape of dataframe after getting rid of the Nan values is", source_df.shape)
    neural = Neural(params)
    """    
    if "index" not in annotated_df.columns.values:
        annotated_df["index"] = annotated_df.index.values
    indices = annotated_df["index"].values

    all_targets =  annotated_df.columns.values

    all_labels = np.transpose(np.array([np.array(annotated_df[target]) for target in all_targets]))
    target_labels = np.transpose(np.array([np.array(annotated_df[target]) for target in targets]))
    #hate_labels = np.transpose(np.array(annotated_df["hate"]))

    init_clock = time.clock()
    anno_docs = [tokenize(sent.lower()) for sent in annotated_df[text_col].values.tolist()]
    print("Tokenizing annotated data took %d seconds " % (time.clock() - init_clock))
    """
    df_text = source_df[params["text_col"]].values.tolist()
    df_text = neural.tokenize_data(df_text)
    neural.learn_vocab(df_text)
    vocab_size = len(neural.vocab)
    X = np.array(neural.tokens_to_ids(df_text))

    y = np.transpose(np.array([np.array(source_df[target].astype(int)) for target in params["target_cols"]]))
    neural.build()

    if params["k_folds"] > 0:
        neural.cv_model(X, y)

    if params["save_vectors"]:
        neural.run_model(X, y, source_df[params["visualize_cols"]])

        # return predictions, parameters, indices
"""
        ######################## Predict the labels for all the data#####################
            results = dict()
            for idx in range((len(anno_ids) // batch_size) + 1):
                text_batch = anno_ids[idx * batch_size: min((idx + 1) * batch_size, len(corpus_ids))]
                lengths = np.array([len(line) for line in text_batch])
                text_batch = lstm.padding(text_batch)
                text_batch = np.array([np.array(line) for line in text_batch])
                if embedding_method == "GloVe":
                    output = sess.run(predictions, feed_dict={train_inputs: text_batch, seq_length: lengths, keep_prob: 1, embedding_placeholder: embeddings})
                else:
                    output = sess.run(predictions, feed_dict={train_inputs: text_batch, seq_length: lengths, keep_prob: 1})
                for target in targets:
                    results.setdefault(target, []).extend(list(np.argmax(output[target], 1)))
            results["index"] = annotated_df["index"].tolist()
            re_dataframe = pd.DataFrame(results)
            #dataframe.to_pickle(data_dir + '/' + "-".join(target for target in targets) + "-outputs-multi.pkl")
            #print("Results saved as " + data_dir + '/' + "-".join(target for target in targets) + "-outputs-multi.pkl")
            for target in targets:
                if target == "hate":
                    annotated_df[target] = pd.to_numeric(annotated_df[target])
                    print("Analyzing results of", target)
                    true = dict()
                    false = dict()
                    all = dict()
                    precision = dict()
                    f1 = dict()
                    recall = dict()
                    checked = list()
                    for i in range(n_outputs):
                        true[i] = 0
                        all[i] = 0
                        false[i] = 0

                    for idx in test_idx:
                        if idx in checked:
                            continue
                        checked.append(idx)
                        if (re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target] == annotated_df.loc[annotated_df["index"] == idx].iloc[0][target]).any():
                            true[re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target]] += 1
                        else:
                            false[re_dataframe.loc[re_dataframe["index"] == idx].iloc[0][target]] +=1
                        all[annotated_df.loc[annotated_df["index"] == idx].iloc[0][target]] += 1

                    for i in range(n_outputs):
                        precision[i] = float(true[i]) / float(true[i] + false[i]) if true[i] + false[i] != 0 else 0
                        recall[i] = float(true[i]) / float(all[i]) if all[i] != 0 else 0
                        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0
                    none.append(f1[0])
                    qes.append(f1[1])
                    prob.append(f1[2])
                    print("none", f1[0], "ques", f1[1], "prob", f1[2])

print(float(sum(none)) / float(len(none)))
print(float(sum(qes)) / float(len(qes)))
print(float(sum(prob)) / float(len(prob)))
print(float(sum(acc)) / float(len(acc)))
results = pd.DataFrame(
    {'none': none,
     'ques': qes,
     'prob': prob,
     'acc': acc
    })
results.to_pickle("results.pkl")
print("fin")
"""