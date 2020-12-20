
# TODO: write "builders" that instantiate an object and call "fit" in appropriate cases

#def tfidf(x, min_df):
    #return TFIDF(min_df=min_df).transform(x).todense().T
#return dmatrices('hd ~ tfidf(body, 0.1)', data, return_type='dataframe')
    """
    for term in patsy_formula.rhs_termlist:
        if len(term.factors) == 0:  # intercept term has no meaning for ntap models
            continue
    """

"""
try:
    if isinstance(data, pd.DataFrame):
        Y = data.loc[:, self.formula['targets']].values
    elif isinstance(data, dict):
        Y = np.array([data[k] for k in self.formula['targets']]).T
except KeyError:
    raise ValueError("\'data\' missing target(s): ",
                     "{}".format(self.formula['targets']))

try:
    for rep_str in self.formula['reps']:
        rep_str = rep_str.strip('(').strip(')')
        transform_model, source_col = rep_str.split('|')
        text = data[source_col]
        if transform_model == 'tfidf':
            X = TFIDF(text).X.transpose()
    #if isinstance(data, pd.DataFrame) or isinstance(data, pd.SparseDataFrame):
except KeyError:
    raise ValueError("\'data\' missing text input(s): ",
                     "{}".format(self.formula['targets']))
    if len(self.formula['predictors']) > 0:
        try:
            X = data.loc[:, self.formula['predictors']]
        except KeyError:
            raise ValueError("\'data\' missing predictors: "
                             "{}".format(' '.join(self.formula['predictors'])))

if Y.shape[1] == 1:
    Y = Y.reshape(Y.shape[0])
"""

