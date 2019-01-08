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

##### Neural Hyperparameters and Architectures

* learning\_rate: default 1e-4
* batch\_size: default 100
* keep\_ratio: dropout ratio, default 0.5
* cell: LSTM
* model: RCNN
* vocab\_size: 

* models: SVM, ElasticNet, LSTM, BiLSTM, CNN, RNN, RCNN
