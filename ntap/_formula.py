from patsy import ModelDesc
from patsy import EvalFactor
from patsy.desc import Term

_VALID_TOKENIZERS = ['words', 'words_nopunc']
_DEFAULT_TOKENIZER = 'words'

def parse_formula(f_str):

    patsy_formula = ModelDesc.from_formula(f_str)

    tokenize = patsy_formula.lhs_termlist

    valid_tokenizers = list()
    for term in tokenize:
        for e in term.factors:
            code = e.code
            if code in _VALID_TOKENIZERS:
                valid_tokenizers.append(code)

    if len(valid_tokenizers) == 0:
        tokenize.insert(0, Term([EvalFactor(_DEFAULT_TOKENIZER)]))
    if len(valid_tokenizers) > 1:
        raise RuntimeError("Multiple tokenizers found in formula\n"
                           f"Specify one from {' '.join(_VALID_TOKENIZERS)}")

    preprocess = [t for t in patsy_formula.rhs_termlist 
                  if len(t.factors) > 0]
    return tokenize, preprocess


# TODO: add "update" syntax to model objects, with the idea of comparing
# models across different settings (formulas). For example, tfidf versus
# BERT, or stemming versus non-stemming

