
import pandas as pd
from ntap.bagofwords import DocTerm, TFIDF, LDA
from ntap.parse import TextPreprocessor
from ntap.supervised import TextRegressor, TextClassifier
#from ntap.embedding import DDR

if __name__ == '__main__':

    df = pd.read_csv("~/PycharmProjects/HateAnnotations/ghc_with_users.tsv", '\t')
    df.body = TextPreprocessor('all').transform(df.body)

    cl = TextClassifier(formula='hd~(tfidf|Text)',
                        model_family='svm')
    cl.fit(data=df)


    #lda_m = LDA(df.body, method='mallet', mallet_path="~/mallet-2.0.8/bin/mallet")

