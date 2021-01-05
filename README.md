# ntap: Neural Text Analysis Pipeline

`ntap` provides a human-readable API for text analysis using modern methods from NLP. 

The main features of `ntap` are:

1. Reduce user interaction with lower-level ML libraries, such as PyTorch or sciki-learn, instead relying on a few high-level interfaces
2. Provide reproducible pipelines for text analyses
3. Define a human-readable grammar/syntax for defining analysis pipelines, using R/S-style formulas
4. Dynamic integration of interpretability research from ML/NLP via `analysis` module

## Installation

`ntap` is available via pip 

``` 
pip install -u ntap
```

Required dependencies will be automatically installed:

* numpy
* pandas
* gensim
* scikit-learn
* tqdm
* patsy
* liwc
* tomotopy

Several packages offer additional functionality, and are optional:

* optuna

```

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




## TODO Features 

```
from ntap.supervised import TextClassifier

# [X+Y] separate models, X+Y multitask
default = TextClassifier(formula='[Harm+Care]~T(text) + speaker.party', data=mftc_df)
bilstm = TextClassifier(formula='Harm ~ encode(text) + speaker.party'},
                        features=[glove_sequences],
                        family='bilstm',  # or ntap.models.bilstm(hidden=[24, 48], ...)
                        data=mftc_df)
```

