from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
import numpy as np
import operator, os, json




def evaluate_models(df, X, targets, lookup_dict, models, seed, feature_methods, scoring_dir, config_text):

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

    scoring_output = os.path.join(scoring_dir, config_text, "-".join(f for f in feature_methods))
    write_evaluations(scoring_dict, scoring_output)
    print("Results written to " + str((scoring_output)))


def elasticnet(X, Y, lookup_dict, scoring_dict, col, kfold, seed):
    print("elastic net model")
    model = "elasticnet"
    scoring_dict[col][model] = dict()
    regressor = SGDRegressor(loss='squared_loss',  # default
                             penalty='elasticnet',
                             tol=1e-4,
                             shuffle=True,
                             random_state=seed,
                             verbose=1
                             )
    choose_regressor = GridSearchCV(regressor, cv=kfold, iid=True,
                                    param_grid={"alpha": 10.0 ** -np.arange(3, 7),
                                                "l1_ratio": np.arange(0.15, 0.20, 0.3)
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
    model = 'gbrt'
    scoring_dict[col][model] = dict()
    gbrt = GradientBoostingRegressor(verbose=2, random_state=seed)
    choose_gbrt = GridSearchCV(gbrt, cv=kfold, 
                    param_grid={"learning_rate": np.arange(.1,1.0,.3),
                                "n_estimators": np.arange(50, 250, 50),
                                "max_depth": np.arange(3,10)})
    choose_gbrt.fit(X, Y)
    best_model = choose_gbrt.best_estimator_
    scoring_dict[col][model]['params'] = choose_gbrt.best_params_
    feature_importance = best_model.feature_importances_
    print(feature_importance)
    return scoring_dict, best_model

                                                
    # Post conditions: best_model is the best model (by CV); scoring_dict[col][model] is updated


def write_evaluations(scoring_dict, scoring_output):

    if not os.path.exists(scoring_output):
        os.makedirs(scoring_output)
    scoring_output = os.path.join(scoring_output, "scores_full" + ".json")
    with open(scoring_output, 'w') as fo:
        json.dump(scoring_dict, fo, indent=4)
