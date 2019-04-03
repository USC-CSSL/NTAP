"""
file: parameters.py
"""

processing = {
              "clean": ["links"], #, "mentions", "hashtags"],
              "preprocess": ["lowercase"]  #["stem"],
             }

features = {"tokenizer": 'wordpunc',
            "stopwords": 'nltk',
            "features": ["dictionary"],  #, "lda", "ddr"],
            "dictionary": 'moral.dic',
            "wordvec": 'glove',
            "vocab_size": 10000,
            "num_topics": 100,
            "num_iter": 500, 
            "ngrams": [0, 1],
            "sent_tokenizer": "tweet"
           }

prediction = {"prediction_task": 'classification',
              "method_string": 'svm',  # svm, elasticnet
              "num_trials": 1,  # run k-fold x-validation num_trials times
              "kfolds": 3
              }

neural = {"learning_rate": 0.0001,
          "batch_size" : 100,
          "keep_ratio" : 0.66,
          "cell": "GRU", #choose from ["GRU", "LSTM"]
          "model": "LSTM_feat",  # choose from ["LSTM", "BiLSTM", "CNN", "RNN", "RCNN"
          "RNN": "BiLSTM",
          "vocab_size": 20000,
          "embedding_size": 300,
          "hidden_size": 100,
          "feature_hidden_size": 100,
          "pretrain": True,
          "train_embedding": False,
          "num_layers": 1,
          "n_outputs": 2,
          "filter_sizes": [2, 3, 4],
          "num_filters": 2,
          "loss": "Mean",  #choose from ["Mean", "Weighted"]
          "save_vectors": False,
          "epochs": 100,
          "word_embedding": "glove",
          "kfolds": 10,
          "random_seed": 55,
          "max_length": 1000,
          "min_length": 2,
          "neural_kfolds": 5,
          "attention_size": 100,
          "train": True,
          "predict": False
}

path = {
    "glove_path": "/Users/neetikasinghal/CSSL/glove.6B/glove.6B.300d.txt", #Set the glove path here
    "word2vec_path": "/Users/neetikasinghal/CSSL/GoogleNews-vectors-negative300.bin.gz", #Set word2vec path here
    "dictionary_path": "/Users/neetikasinghal/CSSL/",
    "mallet_path": "/Users/neetikasinghal/CSSL/mallet-2.0.8/bin/mallet",
    "corenlp_path": "/Users/neetikasinghal/CSSL/stanford-corenlp-full-2018-10-05/",
    "tagme_token": "ec107e88-e1b9-494a-bbc4-00f9e214efd8-843339462"
}

