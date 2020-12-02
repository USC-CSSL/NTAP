import pandas as pd
from ntap.bagofwords import DocTerm, LDA
from ntap.parse import TextPreprocessor

if __name__ == '__main__':

    df = pd.read_csv("~/PycharmProjects/HateAnnotations/ghc_with_users.tsv", '\t')
    df.body = TextPreprocessor('all').transform(df.body)
    dt = DocTerm(df.body)
    print(dt)
    lda_m = LDA(df.body, method='mallet', mallet_path="~/mallet-2.0.8/bin/mallet")
    print(lda_m.model)
