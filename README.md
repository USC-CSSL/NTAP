


# ntap: Neural Text Analysis Pipeline

`ntap` is a python package built on top of `tensorflow`, `sklearn`, `pandas`, `gensim`, `nltk`, and other libraries to facilitate the core functionalities of text analysis using modern methods from NLP. 

## Data loading and Text featurization

All `ntap` functionalities use the Dataset object class, which is responsible for loading datasets from file, cleaning text, transforming text into features, and saving results to file. 

## ntap.data.Dataset
```
Dataset(source, tokenizer="wordpunct", vocab_size=5000, embed="glove",
		min_token=5, stopwords=None, stem=False, lower=True, max_len=100,
		include_nums=False, include_symbols=False, num_topics=100, 
		lda_max_iter=500)
```
### Parameters

* `source`: _str_, path to single data file. Supported formats: newline-delimited `.json`, `.csv`, `.tsv`, saved Pandas DataFrame as `.pkl` file
* `tokenizer`: _str_, select which tokenizer to use. if `None`, will tokenize based on white-space. Options are based on `nltk` word tokenizers: "wordpunct", ... (others not currently supported)
* `vocab_size`: _int_, keep the top `vocab_size` types, by frequency. Used in bag-of-words features, as well as neural methods. If `None`, use all of vocabulary.
* `embed`: _str_, select which word embedding to use for initialization of embedding layer. Currently only `glove` is supported
* `min_token`: _int_, indicates the minimum size, by number of tokens, for a document to be included after calling `clean`. 
* `stopwords`: _iterable_ or _str_, set of words to exclude. Default is `None`, which excludes no words. Options include lists/sets, as well as strings indicating the use of a saved list: `nltk` is the only currently supported option, and indicates the default `nltk` English list
* `stem`: _bool_ or _str_, if `False` then do not stem/lemmatize, otherwise follow the stemming procedure named by `stem`. Options are `snowball`
* `lower`: _bool_, if `True` then cast all alpha characters to lowercase
* `max_len`: _int_, maximum length, by number of valid tokens, for a document to be included during modeling. `None` will result in the maximum length being calculated by the existing document set
* `include_nums`: _bool_, if `True`, then do not discard tokens which contain numeric characters. Examples of this include dates, figures, and other numeric datatypes.
* `include_symbols`: _bool_, if `True`, then do not discard tokens which contain non-alphanumeric symbols
* `num_topics`: _int_, sets default number of topics to use if `lda` method is called at a later point. 
* `lda_max_iter`: _int_, sets default number of iterations of Gibbs sampling to run during LDA model fitting

### Methods

The Dataset class has a number of methods for control over the internal functionality of the class, which are called by Method objects. The most important stand-alone methods are the following:

* `Dataset.set_params(**kwargs)`:
	* Can be called at any time to reset a subset of the parameters in `Dataset`
	* TODO: call specific refitting (i.e. `__learn_vocab`)
* `Dataset.clean(column, remove=["hashtags", "mentions", "links"], mode="remove")`:
	* Removes any tokens (before calling tokenizer) matching the descriptions in the `remove` list. Then tokenizes documents in `column`, defines the vocabulary, the prunes documents from the Dataset instance that do not match the length criteria. All these are defined by the stored parameters in Dataset
	* `column`: _str_, indicates the column name of the text file
	* `remove`: _list_ of _str_, each item indicates a type of token to remove. If `None` or list is empty, no tokens are removed
	* `mode`: _str_, for later iterations, could potentially store hashtag or links. Currently only option is `remove`

The Dataset object supports a number of feature methods (e.g. LDA, TFIDF), which can be called directly by the user, or implicitly during a Method construction (see Method documentation)

* `Dataset.lda(column, method="mallet", save_model=None, load_model=None)`:
	* Uses `gensim` wrapper of `Mallet` java application. Currently only this is supported, though other implementations of LDA can be added. `save_model` and `load_model` are currently unsupported
	* `column`: _str_, text column
	* `method`: only "mallet" is supported
	* `save_model`: _str_, indicate path to save trained topic model. Not yet implemented
	* `load_model`: _str_, indicate path to load trained topic model. Not yet implemented
* `Dataset.ddr(column, dictionary, **kwargs)`:
	* Only method which must be called in advance (currently; advanced versions will store dictionary internally
	* `column`: column in Dataset containing text. Does not have to be tokenized.
	* `dictionary`: _str_, path to dictionary file. Current supported types are `.json` and `.csv`. `.dic` to be added in a later version
	* possible `kwargs` include `embed`, which can be used to set the embedding source (i.e. `embed="word2vec"`, but this feature has not yet been added)
* `Dataset.tfidf(column)`:
	* uses `gensim` TFIDF implementation. If `vocab` has been learned previously, uses that. If not, relearns and computes DocTerm matrix
	* `column`: _str_, text column
* Later methods will include BERT, GLOVE embedding averages

### Examples

Below are a set of use-cases for the Dataset object. Methods like `SVM` are covered elsewhere, and are included here only for illustrative purposes.

```
from ntap.data import Dataset
from ntap.models import RNN, SVM

gab_data = Dataset("./my_data/gab.tsv")
other_gab_data = Dataset("./my_data/gab.tsv", vocab_size=20000, stem="snowball", max_len=1000)
gab_data.clean()
other_gab_data.clean() # using stored parameters
other_gab_data.set_params(include_nums=True) # reset parameter
other_gab_data.clean() # rerun using updated parameters

gab_data.set_params(num_topics=50, lda_max_iter=100)
base_gab = SVM("hate ~ lda(text)", data=gab_data)
base_gab2 = SVM("hate ~ lda(text)", data=other_gab_data)
```

# Base Models

For supervised learning tasks, `ntap` provides two (currently) baseline methods, `SVM` and `LM`. `SVM` uses `sklearn`'s implementation of Support Vector Machine classifier, while `LM` uses either `ElasticNet` (supporting regularized linear regression) or `LinearRegression` from `sklearn`. Both models support the same type of core modeling functions: `CV`, `train`, and `predict`, with `CV` optionally supporting Grid Search.

All methods are created using an `R`-like formula syntax. Base models like `SVM` and `LM` only support single target models, while other models support multiple targets.

## ntap.models.SVM

```
SVM(formula, data, C=1.0, class_weight=None, dual=False, penalty='l2', loss='squared_hinge', tol=0.0001, max_iter=1000, random_state=None)

LM(formula, data, alpha=0.0, l1_ratio=0.5, max_iter=1000, tol=0.001, random_state=None)
```

### Parameters
* formula: _str_, contains a single `~` symbol, separating the left-hand side (the target/dependent variable) from the right-hand side (a series of `+`-delineated text tokens). The right hand side tokens can be either a column in Dataset object given to the constructor, or a feature call in the following form: `<featurename>(<column>)`. 
* `data`: _Dataset_, an existing Dataset instance
* `tol`: _float_, stopping criteria (difference in loss between epochs)
* `max_iter`: _int_, max iterations during training 
* `random_state`: _int_

SVM:
* `C`: _float_, corresponds to the `sklearn` "C" parameter in SVM Classifier
* `dual`: _bool_, corresponds to the `sklearn` "dual" parameter in SVM Classifier
* `penalty`: _string_, regularization function to use, corresponds to the `sklearn` "penalty" parameter
* `loss`: _string_, loss function to use, corresponds to the `sklearn` "loss" parameter

LM: 
* `alpha`: _float_, controls regularization. `alpha=0.0` corresponds to Least Squares regression. `alpha=1.0` is the default ElasticNet setting
* `l1_ratio`: _float_, trade-off between L1 and L2 regularization. If `l1_ratio=1.0` then it is LASSO, if `l1_ratio=0.0` it is Ridge

### Functions

A number of functions are common to both `LM` and `SVM`

* `set_params(**kwargs)`
* `CV`:
	* Cross validation that implicitly support Grid Search. If a list of parameter values (instead of a single value) is given, `CV` runs grid search over all possible combinations of parameters
	* `LM`: `CV(data, num_folds=10, metric="r2", random_state=None)`
	* `SVM`: `CV(data, num_epochs, num_folds=10, stratified=True, metric="accuracy")`
		* `num_epochs`: number of epochs/iterations to train. This should be revised
		* `num_folds`: number of cross folds
		* `stratified`: if true, split data using stratified folds (even split with reference to target variable)
		* `metric`: metric on which to compare different CV results from different parameter grids (if no grid search is specified, no comparison is done and `metric` is disregarded)
	* Returns: An instance of Class `CV_Results`
		* Contains information of all possible classification (or regression) metrics, for each CV fold and the mean across folds
		* Contains saved parameter set 
* `train`
	* Not currently advised for user application. Use `CV` instead
* `predict
	* Not currently advised for user application. Use `CV` instead

### Examples

```
from ntap.data import Dataset
from ntap.models import SVM

data = Dataset("./my_data.csv")
model = SVM("hate ~ tfidf(text)", data=data)
basic_cv_results = model.CV(num_folds=5)
basic_cv_results.summary()
model.set_params(C=[1., .8, .5, .2, .01]) # setting param
grid_searched = model.CV(num_folds=5)
basic_cv_results.summary()
basic_cv_results.params
```

# Models

One basic model has been implemented for `ntap`: `RNN`. Later models will include `CNN` and other neural variants. All model classes (`CNN`, `RNN`, etc.) have the following methods: `CV`, `predict`, and `train`. 

Model formulas using text in a neural architecture should use the following syntax: 
`"<dependent_variable> ~ seq(<text_column>)"`

## `ntap.models.RNN`

```
RNN(formula, data, hidden_size=128, cell="biLSTM", rnn_dropout=0.5, embedding_dropout=None,
	optimizer='adam', learning_rate=0.001, rnn_pooling='last', embedding_source='glove', 
	random_state=None)
```

### Parameters

* `formula`
	* similar to base methods, but supports multiple targets (multi-task learning). The format for this would be: `"hate + moral ~ seq(text)"`
* `data`: _Dataset_ object 
* `hidden_size`: _int_, number of hidden units in the 1-layer RNN-type model\
* `cell`: _str_, type of RNN cell. Default is a bidirectional Long Short-Term Memory (LSTM) cell. Options include `biLSTM`, `LSTM`, `GRU`, and `biGRU` (bidirectional Gated Recurrent Unit)
* `rnn_dropout`: _float_, proportion of parameters in the network to randomly zero-out during dropout, in a layer applied to the outputs of the RNN. If `None`, no dropout is applied (not advised)
* `embedding_dropout`: _str_, not implemented
* `optimizer`: _str_, optimizer to use during training. Options are: `adam`, `sgd`, `momentum`, and `rmsprop`
* `learning_rate`: learning rate during training
* `rnn_pooling`: _str_ or _int_. If _int_, model has self-attention, and a Feed-Forward layer of size `rnn_pooling` is applied to the outputs of the RNN layer in order to produce the attention alphas. If string, possible options are `last` (default RNN behavior, where the last hidden vector is taken as the sentence representation and prior states are removed) `mean` (average hidden states across the entire sequence) and `max` (select the max hidden vector)
* `embedding_source`: _str_, either `glove` or (other not implemented)
* `random_state`: _int_

### Functions

* `CV(data, num_folds, num_epochs, comp='accuracy', model_dir=None)`
	* Automatically performs grid search if multiple values are given for a particular parameter
	* `data`: _Dataset_ on which to perform CV
	* `num_folds`: _int_
	* `comp`: _str_, metric on which to compare different parameter grids (does not apply if no grid search)
	* `model_dir`: if `None`, trained models are saved in a temp directory and then discarded after script exits. Otherwise, `CV` attempts to save each model in the path given by `model_dir`. 
	* Returns: _CV_results_ instance with best model stats (if grid search), and best parameters (not supported)
* `train(data, num_epochs=30, batch_size=256, indices=None, model_path=None)`
	* method called by `CV`, can be called independently. Can train on all data (`indices=None`) or a specified subset. If `model_path` is `None`, does not save model, otherwise attempt to save model at `model_path`
	* `indices`: either `None` (train on all data) or list of _int_, where each value is an index in the range `(0, len(data) - 1)`
* `predict(data, model_path, indices=None, batch_size=256, retrieve=list())`
	* Predicts on new data. Requires a saved model to exist at `model_path`.
	* `indices`: either `None` (train on all data) or list of _int_, where each value is an index in the range `(0, len(data) - 1)`
	* `retrieve`: contains list of strings which indicate which model variables to retrieve during prediction. Includes: `rnn_alpha` (if attention model) and `hidden_states` (any model)
	* Returns: dictionary with {variable_name: value_list}. Contents are predicted values for each target variable and any model variables that are given in `retrieve`.

```
from ntap.data import Dataset
from ntap.models import RNN

data = Dataset("./my_data.csv")
base_lstm = RNN("hate ~ seq(text)", data=data)
attention_lstm = RNN("hate ~ seq(text)", data=data, rnn_pooling=100) # attention
context_lstm = RNN("hate ~ seq(text) + speaker_party", data=data) # categorical variable
base_model.set_params({"hidden"=[200, 50], lr=[0.01, 0.05]}) # enable grid search during CV

# Grid search and print results from best parameters
base_results = base_model.CV()
base_results.summary()

# Train model and save. Predict for 6 specific instances and get alphas
attention_lstm.train(data, model_path="./trained_model")
predictions = attention_lstm.predict(data, model_path="./trained_model",
							indices=[0,1,2,3,4,5], retrieve=["rnn_alphas"])
for alphas in predictions["rnn_alphas"]:
	print(alphas)  # prints list of floats, each the weight of a word in the ith document
```

# Coming soon...

`MIL(formula, data, ...)`
- not implemented
- 
`HAN(formula, data, ...)`
- not implemented

`CNN()`
- not implemented

## `NTAP.data.Tagme`
Not implemented
`Tagme(token="system", p=0.15, tweet=False)`
* `token` (`str`): Personal `Tagme` token. Users can retrieve token by  [Creating Account](https://sobigdata.d4science.org/home?p_p_id=58&p_p_lifecycle=0&p_p_state=maximized&p_p_mode=view&p_p_col_id=column-1&p_p_col_count=2&saveLastPath=false&_58_struts_action=%2Flogin%2Fcreate_account). Default behavior ("system") assumes `Tagme` token has been set during installation of `NTAP`.
Members:
* get_tags(list-like of strings)
	* Stores `abstracts` and `categories` as member variables 
* reset()
* `abstracts`: dictionary of {`entity_id`: `abstract text ...`}
* `categories`: dictionary of {`entity_id`: `[category1, category2, `}
```
data = Dataset("path.csv")
data.tokenize(tokenizer='tweettokenize')
abstracts, categories = data.get_tagme(tagme_token=ntap.tagme_token, p=0.15, tweet=False)
# tagme saved as data object at data.entities
data.background_features(method='pointwise-mi', ...)  # assumes data.tagme is set; creates features
saves features at data.background

background_mod = RNN("purity ~ seq(words) + background", data=data)
background_mod.CV(kfolds=10)
```
## `NTAP.data.TACIT`
not implemented. Wrapper around TACIT instance
`TACIT(path_to_tacit_directory, params to create tacit session)`
