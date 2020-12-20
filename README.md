


# ntap: Neural Text Analysis Pipeline

`ntap` provides a human-readable API for text analysis using modern methods from NLP. 

The goals of `ntap` are:

1. Reduce user interaction with lower-level ML libraries, such as PyTorch or sciki-learn, instead relying on a few high-level interfaces (similarly to R/S)
2. Provide reproducible pipelines for text analyses
3. Define a human-readable grammar/syntax for defining analysis pipelines, using R/S-style formulas
4. Dynamic integration of interpretability research from ML/NLP via `analysis` module

## Pipelines

Instantiate a new analysis via existing template pipelines (LIWC, topic modeling, fine tuning) or load a previous analysis from configuration/model files

## Data input

NTAP tries to enable flexible data input/output, working with `pandas` DataFrames/Series objects as well as python dictionaries and lists. 

```
dataset = pd.DataFrame({'text': ['how now brown cow',
                                 'the tea in Nepal is very hot',
                                 'but the coffee in Peru is much hotter'],
                        'author': ['brendan', 'erin', 'kevin']})
```

## Text cleaning/tokenization -> Implicit

### ntap.parse.TextPreprocessor

Processing scripts are handled by the TextPreprocessor object, which uses formulas to specify cleaning operations

```
proc = TextProcessor('clean-digits')
dataset['text_clean'] = proc.transform(dataset.text)

```
* `include_nums`: _bool_, if `True`, then do not discard tokens which contain numeric characters. Examples of this include dates, figures, and other numeric datatypes.
* `include_symbols`: _bool_, if `True`, then do not discard tokens which contain non-alphanumeric symbols

#### ntap.bagofwords

```
from ntap.bagofwords import DocTerm, TFIDF, LDA
docterm = DocTerm(dataset.clean_text, tokenizer='basic')

### Parameters

* `tokenizer`: _str_, default 'regex' (\w{2,20})
* `vocab_size`: _int_, keep the top vocabulary terms by frequency
* `max_len`: _int_, maximum length, by number of valid tokens, for a document to be included during modeling. `None` will result in the maximum length being calculated by the existing document set
* `num_topics`: _int_, sets default number of topics to use if `lda` method is called at a later point. 
* `lda_max_iter`: _int_, sets default number of iterations of Gibbs sampling to run during LDA model fitting

### Methods

* `min_token`: _int_, indicates the minimum size, by number of tokens, for a document to be included after calling `clean`. 
* `embed`: _str_, select which word embedding to use for initialization of embedding layer. Currently only `glove` is supported
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

## Demonstration of new ntap API

```
#from transformers import BERT

from ntap.supervised import TextClassifier

from ntap.preprocess import TextPreprocessor
from ntap.embeddings import embed_reader, embed_sequences
from ntap.embeddings.ddr import DDR
from ntap.bagofwords import TFIDF, LDA

glove_vecs = embed_reader.glove_from_txt("~/Data/glove.6B/glove.6B.100d.txt")

glove_sequences = embed_sequences(vector_matrix=glove_vecs, max_seq=100)
ddr_mfd2 = DDR(vector_matrix=glove_vecs,
               dic='~/Data/dictionaries/mfd2.dic')


proc = TextPreprocessor()
proc2 = TextPreprocessor('all-numbers') # default without removing numbers


mftc['clean'] = TextPreprocessor.clean(mftc_df['Tweet'])

# multinomial NB and TFIDF (default params)
# syntax: [X+Y] separate models, X+Y multitask
default = TextClassifier(formula='[Harm+Care]~T(text) + speaker.party', data=mftc_df)

svm = TextClassifier(Harm ~ (tfidf|text)', presets={'tfidf':tfidf}, family='svm')
bilstm = TextClassifier(formula='Harm ~ encode(text) + speaker.party'},
                        features=[glove_sequences],
                        family='bilstm',  # or ntap.models.bilstm(hidden=[24, 48], ...)
                        data=mftc_df)

bert_ft = TextClassifier(formula='Harm ~ (bert|text)',
                         family='finetune',  # extra **params
                         data=ghc_df)
```

