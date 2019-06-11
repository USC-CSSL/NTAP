import numpy as np
import json
import os

input_path = "D:/Summer_2019/Computational_Social_Sciences_Lab/cssl/data/baltimore_data.pkl"
glove_path = "D:/Summer_2019/Computational_Social_Sciences_Lab/cssl/data/glove.6B/glove.6B.300d.txt"
word2vec_path = "D:/Summer_2019/Computational_Social_Sciences_Lab/cssl/data/GoogleNews-vectors-negative300.bin.gz"
dictionary_path = "D:/Summer_2019/Computational_Social_Sciences_Lab/cssl/data"
tagme_token =  "ec107e88-e1b9-494a-bbc4-00f9e214efd8-843339462"
jobs = ["clean"]
clean = ["links"]
preprocess = ["lowercase"]
text_col="text"
features = ["dictionary"]
targets = [""]
method = ""
kfolds = 10
stopwords = None
num_words = 10000
sent_tokenizer = "wordpunc"
ngrams = [0, 1]
bom_method = "glove"
num_topics = 50
num_iter = 1000
num_trials = 1
dictionary = "mfd2.0"
random_seed = 123
test_filepath = ""
feature = "dictionary"
method = ""

learning_rates = [0.001, 0.0001]
batch_sizes = [100]
keep_ratios = [0.66]
cells = ["GRU", "LSTM"]
models = ["LSTM", "LSTM_feat", "ATTN", "ATTN_feat", "CNN"]
RNN = "BiLSTM"
vocab_sizes = [10000]
embedding_sizes = [300]
feature_hidden_size = 100
feature_size = 10
pretrain_list = [True]
train_embedding = False
target_cols_set = [["authority","care","fairness","loyalty","purity","moral"]]
hidden_layers_list = [[256,256]]
n_outputs = 2
filter_sizes_list = [[2, 3, 4]]
num_filters_list = [2]
loss = "Mean"
save_vectors_list = False
epochs_list = [100]
word_embedding_list = ["glove"]
kfolds_list = [10]
neural_random_seed = 55
max_length = 1000
min_length = 2
neural_kfolds = 10
attention_size = 100
tasks = ["train","predict"]


path = "./paramFiles/"
if not os.path.isdir(path):
    os.makedirs(path)
count = 1
for task in tasks:
    for pretrain in pretrain_list:
        for filter_sizes in filter_sizes_list:
            for num_filters in num_filters_list:
                for batch_size in batch_sizes:
                    for vocab_size in vocab_sizes:
                        for embedding_size in embedding_sizes:
                            for word_embedding in word_embedding_list:
                                for keep_ratio in keep_ratios:
                                    for model in models:
                                        for cell in cells:
                                            for target_cols in target_cols_set:
                                                for hidden_layers in hidden_layers_list:
                                                    for learning_rate in learning_rates:
                                                        for epochs in epochs_list:
                                                            for kfolds in kfolds_list:
                                                                params = {}
                                                                params["processing"] = {}
                                                                params["processing"]["jobs"] = jobs
                                                                params["processing"]["input_path"] = input_path
                                                                params["processing"]["clean"] = clean
                                                                params["processing"]["preprocess"] = preprocess
                                                                params["processing"]["text_col"] = text_col
                                                                params["baseline"] = {}
                                                                params["baseline"]["features"] = features
                                                                params["baseline"]["targets"] = targets
                                                                params["baseline"]["method"] = method
                                                                params["feature_params"] = {}
                                                                params["feature_params"]["kfolds"] = kfolds
                                                                params["feature_params"]["stopwords"] = stopwords
                                                                params["feature_params"]["num_words"] = num_words
                                                                params["feature_params"]["sent_tokenizer"] = sent_tokenizer
                                                                params["feature_params"]["ngrams"] = ngrams
                                                                params["feature_params"]["bom_method"] = bom_method
                                                                params["feature_params"]["num_topics"] = num_topics
                                                                params["feature_params"]["num_iter"] = num_iter
                                                                params["feature_params"]["num_trials"] = num_trials
                                                                params["feature_params"]["dictionary"] = dictionary
                                                                params["feature_params"]["random_seed"] = random_seed
                                                                params["model"] = {}
                                                                params["model"]["test_filepath"] = test_filepath
                                                                params["model"]["feature"] = feature
                                                                params["model"]["method"] = method
                                                                params["neural_params"] = {}
                                                                params["neural_params"]["learning_rate"] = learning_rate
                                                                params["neural_params"]["batch_size"] = batch_size
                                                                params["neural_params"]["keep_ratio"] = keep_ratio
                                                                params["neural_params"]["cell"] = cell
                                                                params["neural_params"]["model"] = model
                                                                params["neural_params"]["vocab_size"] = vocab_size
                                                                params["neural_params"]["embedding_size"] = embedding_size
                                                                params["neural_params"]["pretrain"] = pretrain
                                                                params["neural_params"]["train_embedding"] = train_embedding
                                                                params["neural_params"]["target_cols"] = target_cols
                                                                params["neural_params"]["hidden_layers"] = hidden_layers
                                                                params["neural_params"]["filter_sizes"] = filter_sizes
                                                                params["neural_params"]["num_filters"] = num_filters
                                                                params["neural_params"]["epochs"] = epochs
                                                                params["neural_params"]["word_embedding"] = word_embedding
                                                                params["neural_params"]["kfolds"] = kfolds
                                                                params["neural_params"]["task"] = task
                                                                params["neural_params"]["feature_hidden_size"] = feature_hidden_size
                                                                params["neural_params"]["feature_size"] = feature_size
                                                                params["neural_params"]["random_seed"] = neural_random_seed
                                                                params["neural_params"]["max_length"] = max_length
                                                                params["neural_params"]["min_length"] = min_length
                                                                params["neural_params"]["neural_kfolds"] = neural_kfolds
                                                                params["neural_params"]["attention_size"] = attention_size
                                                                params["neural_params"]["RNN"] = RNN
                                                                params["neural_params"]["n_outputs"] = n_outputs
                                                                params["neural_params"]["loss"] = loss
                                                                params["neural_params"]["save_vectors_list"] = save_vectors_list
                                                                params["path"] = {}
                                                                params["path"]["glove_path"] = glove_path
                                                                params["path"]["word2vec"] = word2vec_path
                                                                params["path"]["dictionary_path"] = dictionary_path
                                                                params["path"]["tagme_token"] = tagme_token
                                                                filename = "params"+str(count)+".json"
                                                                j = json.dumps(params, indent=4)
                                                                f = open(path+filename, 'w')
                                                                f.write(j)
                                                                f.close()
                                                                count+=1
