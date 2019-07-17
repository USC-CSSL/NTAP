---


---

<h1 id="ntap-neural-text-analysis-pipeline">ntap: Neural Text Analysis Pipeline</h1>
<p><code>ntap</code> is a python package built on top of <code>Tensorflow</code>, <code>sklearn</code>, <code>pandas</code>, and other libraries to facilitate the core functionalities of text analysis using modern methods from NLP.</p>
<h2 id="data-loading-and-text-featurization">Data loading and Text featurization</h2>
<p>All <code>ntap</code> functionalities use the Dataset object class, which is responsible for loading datasets from file, cleaning text, transforming text into features, and saving results to file.</p>
<h2 id="ntap.data.dataset">ntap.data.Dataset</h2>
<pre><code>Dataset(source, tokenizer="wordpunct", vocab_size=5000, embed="glove",
		min_token=5, stopwords=None, stem=False, lower=True, max_len=100,
		include_nums=False, include_symbols=False, num_topics=100, 
		lda_max_iter=500)
</code></pre>
<h3 id="parameters">Parameters</h3>
<ul>
<li><code>source</code>: <em>str</em>, path to single data file. Supported formats: newline-delimited <code>.json</code>, <code>.csv</code>, <code>.tsv</code>, saved Pandas DataFrame as <code>.pkl</code> file</li>
<li><code>tokenizer</code>: <em>str</em>, select which tokenizer to use. if <code>None</code>, will tokenize based on white-space. Options are based on <code>nltk</code> word tokenizers: “wordpunct”, … (others not currently supported)</li>
<li><code>vocab_size</code>: <em>int</em>, keep the top <code>vocab_size</code> types, by frequency. Used in bag-of-words features, as well as neural methods. If <code>None</code>, use all of vocabulary.</li>
<li><code>embed</code>: <em>str</em>, select which word embedding to use for initialization of embedding layer. Currently only <code>glove</code> is supported</li>
<li><code>min_token</code>: <em>int</em>, indicates the minimum size, by number of tokens, for a document to be included after calling <code>clean</code>.</li>
<li><code>stopwords</code>: <em>iterable</em> or <em>str</em>, set of words to exclude. Default is <code>None</code>, which excludes no words. Options include lists/sets, as well as strings indicating the use of a saved list: <code>nltk</code> is the only currently supported option, and indicates the default <code>nltk</code> English list</li>
<li><code>stem</code>: <em>bool</em> or <em>str</em>, if <code>False</code> then do not stem/lemmatize, otherwise follow the stemming procedure named by <code>stem</code>. Options are <code>snowball</code></li>
<li><code>lower</code>: <em>bool</em>, if <code>True</code> then cast all alpha characters to lowercase</li>
<li><code>max_len</code>: <em>int</em>, maximum length, by number of valid tokens, for a document to be included during modeling. <code>None</code> will result in the maximum length being calculated by the existing document set</li>
<li><code>include_nums</code>: <em>bool</em>, if <code>True</code>, then do not discard tokens which contain numeric characters. Examples of this include dates, figures, and other numeric datatypes.</li>
<li><code>include_symbols</code>: <em>bool</em>, if <code>True</code>, then do not discard tokens which contain non-alphanumeric symbols</li>
<li><code>num_topics</code>: <em>int</em>, sets default number of topics to use if <code>lda</code> method is called at a later point.</li>
<li><code>lda_max_iter</code>: <em>int</em>, sets default number of iterations of Gibbs sampling to run during LDA model fitting</li>
</ul>
<h3 id="methods">Methods</h3>
<p>The Dataset class has a number of methods for control over the internal functionality of the class, which are called by Method objects. The most important stand-alone methods are the following:</p>
<ul>
<li><code>Dataset.set_params(**kwargs)</code>:
<ul>
<li>Can be called at any time to reset a subset of the parameters in <code>Dataset</code></li>
<li>TODO: call specific refitting (i.e. <code>__learn_vocab</code>)</li>
</ul>
</li>
<li><code>Dataset.clean(column, remove=["hashtags", "mentions", "links"], mode="remove")</code>:
<ul>
<li>Removes any tokens (before calling tokenizer) matching the descriptions in the <code>remove</code> list. Then tokenizes documents in <code>column</code>, defines the vocabulary, the prunes documents from the Dataset instance that do not match the length criteria. All these are defined by the stored parameters in Dataset</li>
<li><code>column</code>: <em>str</em>, indicates the column name of the text file</li>
<li><code>remove</code>: <em>list</em> of <em>str</em>, each item indicates a type of token to remove. If <code>None</code> or list is empty, no tokens are removed</li>
<li><code>mode</code>: <em>str</em>, for later iterations, could potentially store hashtag or links. Currently only option is <code>remove</code></li>
</ul>
</li>
</ul>
<p>The Dataset object supports a number of feature methods (e.g. LDA, TFIDF), which can be called directly by the user, or implicitly during a Method construction (see Method documentation)</p>
<ul>
<li><code>Dataset.lda(column, method="mallet", save_model=None, load_model=None)</code>:
<ul>
<li>Uses <code>gensim</code> wrapper of <code>Mallet</code> java application. Currently only this is supported, though other implementations of LDA can be added. <code>save_model</code> and <code>load_model</code> are currently unsupported</li>
<li><code>column</code>: <em>str</em>, text column</li>
<li><code>method</code>: only “mallet” is supported</li>
<li><code>save_model</code>: <em>str</em>, indicate path to save trained topic model. Not yet implemented</li>
<li><code>load_model</code>: <em>str</em>, indicate path to load trained topic model. Not yet implemented</li>
</ul>
</li>
<li><code>Dataset.ddr(column, dictionary, **kwargs)</code>:
<ul>
<li>Only method which must be called in advance (currently; advanced versions will store dictionary internally</li>
<li><code>column</code>: column in Dataset containing text. Does not have to be tokenized.</li>
<li><code>dictionary</code>: <em>str</em>, path to dictionary file. Current supported types are <code>.json</code> and <code>.csv</code>. <code>.dic</code> to be added in a later version</li>
<li>possible <code>kwargs</code> include <code>embed</code>, which can be used to set the embedding source (i.e. <code>embed="word2vec"</code>, but this feature has not yet been added)</li>
</ul>
</li>
<li><code>Dataset.tfidf(column)</code>:
<ul>
<li>uses <code>gensim</code> TFIDF implementation. If <code>vocab</code> has been learned previously, uses that. If not, relearns and computes DocTerm matrix</li>
<li><code>column</code>: <em>str</em>, text column</li>
</ul>
</li>
<li>Later methods will include BERT, GLOVE embedding averages</li>
</ul>
<h3 id="examples">Examples</h3>
<p>Below are a set of use-cases for the Dataset object. Methods like <code>SVM</code> are covered elsewhere, and are included here only for illustrative purposes.</p>
<pre><code>from ntap.data import Dataset
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
</code></pre>
<h1 id="base-models">Base Models</h1>
<p>For supervised learning tasks, <code>ntap</code> provides two (currently) baseline methods, <code>SVM</code> and <code>LM</code>. <code>SVM</code> uses <code>sklearn</code>'s implementation of Support Vector Machine classifier, while <code>LM</code> uses either <code>ElasticNet</code> (supporting regularized linear regression) or <code>LinearRegression</code> from <code>sklearn</code>. Both models support the same type of core modeling functions: <code>CV</code>, <code>train</code>, and <code>predict</code>, with <code>CV</code> optionally supporting Grid Search.</p>
<p>All methods are created using an <code>R</code>-like formula syntax. Base models like <code>SVM</code> and <code>LM</code> only support single target models, while other models support multiple targets.</p>
<h2 id="ntap.models.svm">ntap.models.SVM</h2>
<pre><code>SVM(formula, data, C=1.0, class_weight=None, dual=False, penalty='l2', loss='squared_hinge', tol=0.0001, max_iter=1000, random_state=None)

LM(formula, data, alpha=0.0, l1_ratio=0.5, max_iter=1000, tol=0.001, random_state=None)
</code></pre>
<h3 id="parameters-1">Parameters</h3>
<ul>
<li>formula: <em>str</em>, contains a single <code>~</code> symbol, separating the left-hand side (the target/dependent variable) from the right-hand side (a series of <code>+</code>-delineated text tokens). The right hand side tokens can be either a column in Dataset object given to the constructor, or a feature call in the following form: <code>&lt;featurename&gt;(&lt;column&gt;)</code>.</li>
<li><code>data</code>: <em>Dataset</em>, an existing Dataset instance</li>
<li><code>tol</code>: <em>float</em>, stopping criteria (difference in loss between epochs)</li>
<li><code>max_iter</code>: <em>int</em>, max iterations during training</li>
<li><code>random_state</code>: <em>int</em></li>
</ul>
<p>SVM:</p>
<ul>
<li><code>C</code>: <em>float</em>, corresponds to the <code>sklearn</code> “C” parameter in SVM Classifier</li>
<li><code>dual</code>: <em>bool</em>, corresponds to the <code>sklearn</code> “dual” parameter in SVM Classifier</li>
<li><code>penalty</code>: <em>string</em>, regularization function to use, corresponds to the <code>sklearn</code> “penalty” parameter</li>
<li><code>loss</code>: <em>string</em>, loss function to use, corresponds to the <code>sklearn</code> “loss” parameter</li>
</ul>
<p>LM:</p>
<ul>
<li><code>alpha</code>: <em>float</em>, controls regularization. <code>alpha=0.0</code> corresponds to Least Squares regression. <code>alpha=1.0</code> is the default ElasticNet setting</li>
<li><code>l1_ratio</code>: <em>float</em>, trade-off between L1 and L2 regularization. If <code>l1_ratio=1.0</code> then it is LASSO, if <code>l1_ratio=0.0</code> it is Ridge</li>
</ul>
<h3 id="functions">Functions</h3>
<p>A number of functions are common to both <code>LM</code> and <code>SVM</code></p>
<ul>
<li><code>set_params(**kwargs)</code></li>
<li><code>CV</code>:
<ul>
<li>Cross validation that implicitly support Grid Search. If a list of parameter values (instead of a single value) is given, <code>CV</code> runs grid search over all possible combinations of parameters</li>
<li><code>LM</code>: <code>CV(data, num_folds=10, metric="r2", random_state=None)</code></li>
<li><code>SVM</code>: <code>CV(data, num_epochs, num_folds=10, stratified=True, metric="accuracy")</code>
<ul>
<li><code>num_epochs</code>: number of epochs/iterations to train. This should be revised</li>
<li><code>num_folds</code>: number of cross folds</li>
<li><code>stratified</code>: if true, split data using stratified folds (even split with reference to target variable)</li>
<li><code>metric</code>: metric on which to compare different CV results from different parameter grids (if no grid search is specified, no comparison is done and <code>metric</code> is disregarded)</li>
</ul>
</li>
<li>Returns: An instance of Class <code>CV_Results</code>
<ul>
<li>Contains information of all possible classification (or regression) metrics, for each CV fold and the mean across folds</li>
<li>Contains saved parameter set</li>
</ul>
</li>
</ul>
</li>
<li><code>train</code>
<ul>
<li>Not currently advised for user application. Use <code>CV</code> instead</li>
</ul>
</li>
<li>`predict
<ul>
<li>Not currently advised for user application. Use <code>CV</code> instead</li>
</ul>
</li>
</ul>
<h3 id="examples-1">Examples</h3>
<pre><code>from ntap.data import Dataset
from ntap.models import SVM

data = Dataset("./my_data.csv")
model = SVM("hate ~ tfidf(text)", data=data)
basic_cv_results = model.CV(num_folds=5)
basic_cv_results.summary()
model.set_params(C=[1., .8, .5, .2, .01]) # setting param
grid_searched = model.CV(num_folds=5)
basic_cv_results.summary()
basic_cv_results.params
</code></pre>
<h1 id="models">Models</h1>
<p>One basic model has been implemented for <code>ntap</code>: <code>RNN</code>. Later models will include <code>CNN</code> and other neural variants. All model classes (<code>CNN</code>, <code>RNN</code>, etc.) have the following methods: <code>CV</code>, <code>predict</code>, and <code>train</code>.</p>
<p>Model formulas using text in a neural architecture should use the following syntax:<br>
<code>"&lt;dependent_variable&gt; ~ seq(&lt;text_column&gt;)"</code></p>
<h2 id="ntap.models.rnn"><code>ntap.models.RNN</code></h2>
<pre><code>RNN(formula, data, hidden_size=128, cell="biLSTM", rnn_dropout=0.5, embedding_dropout=None,
	optimizer='adam', learning_rate=0.001, rnn_pooling='last', embedding_source='glove', 
	random_state=None)
</code></pre>
<h3 id="parameters-2">Parameters</h3>
<ul>
<li><code>formula</code>
<ul>
<li>similar to base methods, but supports multiple targets (multi-task learning). The format for this would be: <code>"hate + moral ~ seq(text)"</code></li>
</ul>
</li>
<li><code>data</code>: <em>Dataset</em> object</li>
<li><code>hidden_size</code>: <em>int</em>, number of hidden units in the 1-layer RNN-type model\</li>
<li><code>cell</code>: <em>str</em>, type of RNN cell. Default is a bidirectional Long Short-Term Memory (LSTM) cell. Options include <code>biLSTM</code>, <code>LSTM</code>, <code>GRU</code>, and <code>biGRU</code> (bidirectional Gated Recurrent Unit)</li>
<li><code>rnn_dropout</code>: <em>float</em>, proportion of parameters in the network to randomly zero-out during dropout, in a layer applied to the outputs of the RNN. If <code>None</code>, no dropout is applied (not advised)</li>
<li><code>embedding_dropout</code>: <em>str</em>, not implemented</li>
<li><code>optimizer</code>: <em>str</em>, optimizer to use during training. Options are: <code>adam</code>, <code>sgd</code>, <code>momentum</code>, and <code>rmsprop</code></li>
<li><code>learning_rate</code>: learning rate during training</li>
<li><code>rnn_pooling</code>: <em>str</em> or <em>int</em>. If <em>int</em>, model has self-attention, and a Feed-Forward layer of size <code>rnn_pooling</code> is applied to the outputs of the RNN layer in order to produce the attention alphas. If string, possible options are <code>last</code> (default RNN behavior, where the last hidden vector is taken as the sentence representation and prior states are removed) <code>mean</code> (average hidden states across the entire sequence) and <code>max</code> (select the max hidden vector)</li>
<li><code>embedding_source</code>: <em>str</em>, either <code>glove</code> or (other not implemented)</li>
<li><code>random_state</code>: <em>int</em></li>
</ul>
<h3 id="functions-1">Functions</h3>
<ul>
<li><code>CV(data, num_folds, num_epochs, comp='accuracy', model_dir=None)</code>
<ul>
<li>Automatically performs grid search if multiple values are given for a particular parameter</li>
<li><code>data</code>: <em>Dataset</em> on which to perform CV</li>
<li><code>num_folds</code>: <em>int</em></li>
<li><code>comp</code>: <em>str</em>, metric on which to compare different parameter grids (does not apply if no grid search)</li>
<li><code>model_dir</code>: if <code>None</code>, trained models are saved in a temp directory and then discarded after script exits. Otherwise, <code>CV</code> attempts to save each model in the path given by <code>model_dir</code>.</li>
<li>Returns: <em>CV_results</em> instance with best model stats (if grid search), and best parameters (not supported)</li>
</ul>
</li>
<li><code>train(data, num_epochs=30, batch_size=256, indices=None, model_path=None)</code>
<ul>
<li>method called by <code>CV</code>, can be called independently. Can train on all data (<code>indices=None</code>) or a specified subset. If <code>model_path</code> is <code>None</code>, does not save model, otherwise attempt to save model at <code>model_path</code></li>
<li><code>indices</code>: either <code>None</code> (train on all data) or list of <em>int</em>, where each value is an index in the range <code>(0, len(data) - 1)</code></li>
</ul>
</li>
<li><code>predict(data, model_path, indices=None, batch_size=256, retrieve=list())</code>
<ul>
<li>Predicts on new data. Requires a saved model to exist at <code>model_path</code>.</li>
<li><code>indices</code>: either <code>None</code> (train on all data) or list of <em>int</em>, where each value is an index in the range <code>(0, len(data) - 1)</code></li>
<li><code>retrieve</code>: contains list of strings which indicate which model variables to retrieve during prediction. Includes: <code>rnn_alpha</code> (if attention model) and <code>hidden_states</code> (any model)</li>
<li>Returns: dictionary with {variable_name: value_list}. Contents are predicted values for each target variable and any model variables that are given in <code>retrieve</code>.</li>
</ul>
</li>
</ul>
<pre><code>from ntap.data import Dataset
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
</code></pre>
<h1 id="coming-soon...">Coming soon…</h1>
<p><code>MIL(formula, data, ...)</code></p>
<ul>
<li>not implemented</li>
<li></li>
</ul>
<p><code>HAN(formula, data, ...)</code></p>
<ul>
<li>not implemented</li>
</ul>
<p><code>CNN()</code></p>
<ul>
<li>not implemented</li>
</ul>
<h2 id="ntap.data.tagme"><code>NTAP.data.Tagme</code></h2>
<p>Not implemented<br>
<code>Tagme(token="system", p=0.15, tweet=False)</code></p>
<ul>
<li><code>token</code> (<code>str</code>): Personal <code>Tagme</code> token. Users can retrieve token by  <a href="https://sobigdata.d4science.org/home?p_p_id=58&amp;p_p_lifecycle=0&amp;p_p_state=maximized&amp;p_p_mode=view&amp;p_p_col_id=column-1&amp;p_p_col_count=2&amp;saveLastPath=false&amp;_58_struts_action=%2Flogin%2Fcreate_account">Creating Account</a>. Default behavior (“system”) assumes <code>Tagme</code> token has been set during installation of <code>NTAP</code>.<br>
Members:</li>
<li>get_tags(list-like of strings)
<ul>
<li>Stores <code>abstracts</code> and <code>categories</code> as member variables</li>
</ul>
</li>
<li>reset()</li>
<li><code>abstracts</code>: dictionary of {<code>entity_id</code>: <code>abstract text ...</code>}</li>
<li><code>categories</code>: dictionary of {<code>entity_id</code>: <code>[category1, category2,</code>}</li>
</ul>
<pre><code>data = Dataset("path.csv")
data.tokenize(tokenizer='tweettokenize')
abstracts, categories = data.get_tagme(tagme_token=ntap.tagme_token, p=0.15, tweet=False)
# tagme saved as data object at data.entities
data.background_features(method='pointwise-mi', ...)  # assumes data.tagme is set; creates features
saves features at data.background

background_mod = RNN("purity ~ seq(words) + background", data=data)
background_mod.CV(kfolds=10)
</code></pre>
<h2 id="ntap.data.tacit"><code>NTAP.data.TACIT</code></h2>
<p>not implemented. Wrapper around TACIT instance<br>
<code>TACIT(path_to_tacit_directory, params to create tacit session)</code></p>

