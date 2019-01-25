## Dynamic Analysis of Text with Advanced Methods (DATAM)

A code repository for a text analysis pipeline integrating cutting edge NLP methods with interpretable baselines

### Overview

This pipeline is for the wider application of advanced methodologies for text analysis. In terms of software, it heavily uses python packages _sklearn_ and _tensorflow_ for the development of established and cutting-edge machine learning methods, respectively. 

### Installation

1. DATAM requires python3.(4-6) to be installed [download 3.6](https://www.python.org/downloads/release/python-367/)
2. It is recommended to use a virtual environment to manage python libraries and dependencies (but not required)
To install with pip:
```$ pip install virtualenv```
or
```$ sudo pip install virtualenv```

Set up a virtualenv environment and install packages from `requirements.txt` (or `requirements-gpu.txt`):
```
$ virtualenv myenv -p path/to/python/interpreter
$ source myenv/bin/activate
$ pip install -r requirements.txt
```


### External Data

DATAM makes use of a number of external resources, such as word vectors and Stanford's CoreNLP. Download them first, and set the appropriate environment variables (see below)

1. Word2vec [download](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)
Set environment variable (bash):
		```
		export WORD2VEC_PATH=path/to/GoogleNews-vectors-negative300.bin.gz
		```
2. GloVe Vectors [download](https://nlp.stanford.edu/projects/glove/)
		```
		export GLOVE_PATH=path/to/glovefile.txt
		```
3. CoreNLP [download](https://stanfordnlp.github.io/CoreNLP/download.html)
		```
		export CORENLP=path/to/stanford-corenlp-full-YYYY-MM-DD/
		```
4. Dictionaries
Set up a directory to contain any directories you want to use in DATAM, such as Moral Foundations Dictionary (MFD) or LIWC categories. 
        ```
        export DICTIONARIES=path/to/dictionaries/directory/
        ```
5. Access Keys
To use the entity-tagging and linking system (provided by [Tagme](https://tagme.d4science.org/tagme/)) sign up for the service and set up your access key:
		```
		export TAGME="<my_access_key"
		```

### Processing Pipeline

This component takes raw text as input and produces data ready for entry into either baseline or other machine learning methods. 

* cleaning
Functionality includes text cleaning, 

### Baseline Pipeline

The baseline implements methods which separately generate features for supervised models. 

Implemented feature methods:

* TFIDF
* LDA
* Dictionary (Word Count)
* Distributed Dictionary Representations
* Bag of Means (averaged word embeddings)
	* Word2Vec (skipgram)
	* Glove (300-d trained on Wikipedia)
	* FastText (currently not supported)

Given the prediction task (classification or regression) two baseline methods are implemented:

* SVM classification
* ElasticNet Regression


### Methods Pipeline


### Evaluation and Analysis Pipeline



END MAIN DOCUMENT
###### Parameter Guide (deprecated)

All modularity for DATAM is handled by changing parameters, which can be accessed by changing the `parameters.py` file in the main directory. There are several categories of parameters, each used for distinct parts of the pipeline: `processing`, `features`, `prediction`, and `neural`.

##### processing

* extract (list): types of data to extract from text. Options are `link`, `mentions`, `hashtag`, `emojis`.
* preprocess (list): text processing steps to take. Options include `stem` and `lowercase`.

##### features

* tokenizer (str): identifier of tokenizer to use. Options are `happier` (the happier tokenizer for tweets), `wordpunc` for the wordpunc tokenizer from NLTK, and `tweet` for the tweet tokenizer. 
* stopwords (str): identifier for the stopword list to use. Options are `nltk`, `sklearn`, and None (to not remove stopwords.
* features (list): list of feature generators to use. Options include `tfidf`, `lda`, `ddr`, `bagofmeans`, and `dictionary`. 
* dictionary(str): dictionary to use for either the `dictionary` or the `ddr` method. Options include `mfd` and `liwc` (not yet implemented). 
* wordvec(str): word embedding vectors to use. Options include `glove`, `word2vec`, and `fasttext`.
* vocab_size(int): number of top-words to use in building vocabulary
* num_topics(int): number of topics to use for LDA
* num_iter(int): number of iterations to train LDA
* ngrams([lower, upper]): range of n-grams to use for building vobulary

##### prediction

* prediction_task(str): options are `classification` and `regression`
* method_type(str): options are `baseline`, `entitylinking`, `dnn`
* method_string(str): options are `svm`, `elasticnet`, `lstm`, `bilstm`, `cnn`, `rnn`, `rcnn`
* num_trials(int): how many cross-validations/trials to run. 
* kfolds(int): how many folds to use for k-fold cross-validation

##### neural 
3. Set GPU flag
If you want to use GPU: set `USE_GPU=true` in file `set-environ.sh`
TBD
