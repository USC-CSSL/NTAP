## Neural Text Analysis Pipeline (NTAP)


(Project is currently in pre-release phase)

A python-based pipeline for applying neural methods to text analysis.

### Overview

This pipeline is for the wider application of advanced methodologies for text analysis. It uses python packages _sklearn_, _gensim_, and _tensorflow_ for the development of established and cutting-edge machine learning methods, respectively. 

### Installation

1. NTAP requires python3.(4-6) to be installed [download 3.6](https://www.python.org/downloads/release/python-367/)
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

NTAP makes use of a number of external resources, such as word vectors and Stanford's CoreNLP. Download them first, and set the appropriate environment variables (see below)

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
Set up a directory to contain any directories you want to use in NTAP, such as Moral Foundations Dictionary (MFD) or LIWC categories. 
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



