
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


proc = TextPreprocessor(formula='.')  # default sequence of cleaning transforms
proc2 = TextPreprocessor(formula='-numbers') # default without removing numbers


mftc['clean'] = TextPreprocessor.clean(mftc_df['Tweet'])

# multinomial NB and TFIDF (default params)
# syntax: [X+Y] separate models, X+Y multitask
default = TextClassifier(formula='[Harm+Care]~T(text) + speaker.party', data=mftc_df)

svm = TextClassifier(formula='Harm ~ encode(text) + speaker.party'},
                     features=ddr_transformer,  # list corresponds to multiple encodes...
                     family='svm',
                     data=mftc_df)
bilstm = TextClassifier(formula='Harm ~ encode(text) + speaker.party'},
                        features=[glove_sequences],
                        family='bilstm',  # or ntap.models.bilstm(hidden=[24, 48], ...)
                        data=mftc_df)

bert_ft = TextClassifier(formula='Harm ~ encode(text)',
                          features=[pretrained_bert_model],
                          family='finetune',  # extra **params
                          data=ghc_df)

