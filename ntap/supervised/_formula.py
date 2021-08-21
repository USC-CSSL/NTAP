from patsy import dmatrices, ModelDesc, INTERCEPT

# TODO: write "builders" that instantiate an object and call "fit" in appropriate cases

def _parse_formula(formula_str, include_intercept=False):
    """ Wrap some extra functionality into Patsy formula parse """

    _form = ModelDesc.from_formula(formula_str)

    # patsy (by default) includes intercept. Discard this on RHS
    if not include_intercept:
        _form.rhs_termlist = [t for t in _form.rhs_termlist 
                              if len(t.factors) != INTERCEPT]

    #print(_form.lhs_termlist)
    #_categoricals = [t for t in _form.lhs_termlist 
                    #if t.startswith("C(") and t.endswith(")")]
    #_task = 'classify' if len(categoricals) > 0 else 'regress'

    #if len(_categoricals) != len(_form.lhs_termlist):
        #raise ValueError(f"Mixed targets detected in {formula_str}. "
                         #"Specify all categoricals "
                         #"using the C(...) syntax or all continuous.")

    #_num_classes = None if _task == 'classify' else len(_form.lhs_termlist)
    _num_classes = 2
    _task = 'classify'

    return _form, _task, _num_classes


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

