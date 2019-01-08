"""
file: parameters.py
"""

processing = {
              "extract": ["link", "mentions", "hashtag"],
              "preprocess": ["lowercase"]  #["stem"],
             }

features = {"tokenizer": 'wordpunc',
            "stopwords": 'nltk',
            "features": ["ddr"],  #, "lda", "ddr"],
            "dictionary": 'mfd',
            "wordvec": 'glove',
            "vocab_size": 10000,
            "num_topics": 100,
            "num_iter": 500, 
            "ngrams": [0, 1]
           }

prediction = {"prediction_task": 'classification',
              "method_type": 'baseline',  # baseline, entitylinking, dnn
              "method_string": 'svm',  # svm, elasticnet
              "num_trials": 1,  # run k-fold x-validation num_trials times
              "kfolds": 3
              }

neural = {"learning_rate": 0.0001,
          "batch_size" : 100,
          "keep_ratio" : 0.66,
          "cell" : "LSTM", #choose from ["GRU", "LSTM"]
          "model" : "RCNN",  # choose from ["LSTM", "BiLSTM", "CNN", "RNN", "RCNN"
          "vocab_size" : 10000,
          "embedding_size" : 300,
          "hidden_size" : 256,
          "pretrain" : True,
          "train_embedding" : False,
          "num_layers" : 1,
          "n_outputs" : 3,
          "filter_sizes" : [2, 3, 4],
          "num_filters" : 2,
          "loss" : "Mean",  #choose from ["Mean", "Weighted"]
          "save_vectors" : False,
          "epochs" : 500,
}

