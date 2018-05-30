from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
import numpy as np
import operator




def evaluate_models(df, X, targets, lookup_dict, models, seed):

    scoring_dict = dict()

    for col in targets:
        print("Working on predicting {}".format(col))
        scoring_dict[col] = dict()
        Y = df[col].values.tolist()
        kfold = model_selection.KFold(n_splits=3, random_state=seed)
        for model in models:
            scoring_dict, best_model = globals()[model](X, Y, lookup_dict, scoring_dict, col, kfold, seed)

        for metric in ['r2']:
            results = best_model.score(X, Y)
            scoring_dict[col][model][metric + "_mean"] = "{0:.3f}".format(results.mean())
            scoring_dict[col][model][metric + "_std"] = "{0:.3f}".format(results.std())

    return scoring_dict


def elasticnet(X, Y, lookup_dict, scoring_dict, col, kfold, seed):
    print("elastic net model")
    model = "elasticnet"
    scoring_dict[col][model] = dict()
    regressor = SGDRegressor(loss='squared_loss',  # default
                             penalty='elasticnet',
                             max_iter=50,
                             shuffle=True,
                             random_state=seed,
                             verbose=0
                             )
    choose_regressor = GridSearchCV(regressor, cv=kfold, iid=True,
                                    param_grid={"alpha": 10.0 ** -np.arange(1, 7),
                                                "l1_ratio": np.arange(0.15, 0.25, 0.05)
                                                }
                                    )

    choose_regressor.fit(X, Y)
    best_model = choose_regressor.best_estimator_
    scoring_dict[col][model]['params'] = choose_regressor.best_params_
    coef_dict = {i: val for i, val in enumerate(best_model.coef_)}
    word_coefs = {lookup_dict[i]: val for i, val in coef_dict.items()}
    abs_val_coefs = {word: abs(val) for word, val in word_coefs.items()}
    top_features = sorted(abs_val_coefs.items(), key=operator.itemgetter(1), reverse=True)[:100]
    real_weights = [[word, word_coefs[word]] for word, _ in top_features]
    scoring_dict[col][model]['top_features'] = real_weights

    return scoring_dict, best_model

def GBRT(X, Y, lookup_dict, scoring_dict, col, kfold, seed):
    print("GBRT model")
    # Do GBRT shit
    # Post conditions: best_model is the best model (by CV); scoring_dict[col][model] is updated
