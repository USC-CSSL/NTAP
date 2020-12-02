name = "ntap"
from . import parse
from . import supervised #import TextClassifier, TextRegressor
from .ops import cross_validate, train, predict
from . import bagofwords #import DocTerm, LDA
from . import embedding #import GloVe

#__all__ = ["TextPreprocessor","TextClassifier","TextRegressor","DocTerm", "LDA", "Glove", "cv", "train", "predict", 'supervised']
