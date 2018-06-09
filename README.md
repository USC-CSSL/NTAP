## Neural-Language-Profiles

Python >= v3.5.2 dependencies:  
* nltk v3.3 (download all corpora)  
* pandas v0.22.0  
* numpy v1.14.0  
* unidecode v1.0.22  
* sklearn v0.20.dev0  
* scipy v1.0.0  
* textblob v0.15.1  
* sklearn-pandas v1.6.0  
* fasttext v0.8.22 (go to the [github repo for fasttext](https://github.com/facebookresearch/fastText#building-fasttext-for-python)) and follow instructions for "Building fastText for Python") 
This project uses the latest release of sklearn, which is in dev-mode. To download & install (requires cython) follow the instructions under "Retrieving the latest code" [here](http://scikit-learn.org/stable/developers/contributing.html#git-repo)


### Parameter Guide:

* "data\_dir": path to data root directory  
* "scoring\_dir": path to scoring directory   
* "config\_text"
  - "indiv": each document is an individual post  
  - "concat": each document is a user's concatenated posts  
* "dataframe\_name": name of input dataframe (not including path). Example: full\_dataframe.pkl  
* "models": list of methods to use
  - "elasticnet"  
  - "GBRT" (in progress: don't use) 
* "targets": list of columns in input dataframe to predict (Y)  
* "metrics": metrics to report in scoring
  - r2  
  - neg\_mean\_squared\_error 
  - neg\_mean\_absolute\_error  
* "random\_seed": random seed to use for experiments  
##### Text Features: Generation and Loading from External Source
* "feature\_method": if generating features from text (and not loading them), specify the method
  - bag-of-means (using word2vec or GloVe)  
  - load\_features  
  - ddr  
  - fasttext (ETA: June 4-9)  
  - infersent (ETA: June 4-9)  
* "text\_col": if generating features, specify text column with a string; otherwise, specify list of feature columns  
* "ordinal\_features": List of features (non-text) which are to be loaded as ordinal  
* "categorical\_features": List of features (non-text) which are to be load as categorical (one-hot) 
##### Word Embedding and DDR Methods
* "training\_corpus"  
  - "GoogleNews": 3 Billion word corpus (skipgram)  
  - "common\_crawl": common crawl corpus (GloVe)  
  - "wiki\_gigaword": Wiki 2014 plus Gigaword (GloVe)  
* "embedding\_method"  
  - skipgram  
  - GloVe (case sensitive)  
* "dictionary": specify dictionary to use if using DDR
  - liwc: Linguistic Inquiry and Word Count
  - mfd: Moral Foundations Dictionary

##### Sentence embedding methods
* "feature\_methods"
  - fasttext
  - infersent
