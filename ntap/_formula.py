from patsy import ModelDesc
from patsy import EvalFactor
from patsy.desc import Term

def parse_formula(f_str):

    patsy_formula = ModelDesc.from_formula(f_str)

    tokenize = patsy_formula.lhs_termlist

    if len(tokenize) == 0:
        tokenize.append(Term([EvalFactor('word')]))

    preprocess = [t for t in patsy_formula.rhs_termlist 
                  if len(t.factors) > 0]
    return tokenize, preprocess


# TODO: add "update" syntax to model objects, with the idea of comparing
# models across different settings (formulas). For example, tfidf versus
# BERT, or stemming versus non-stemming

