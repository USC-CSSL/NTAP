from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
metric_mapping = {'f1': f1_score, 'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score}

import numpy as np
import operator, os, json

def evaluate_models(df, X, targets, lookup_dict, models, seed, feature_methods, scoring_dir, config_text, metrics, output_name):

    scoring_dict = dict()
    

    for col in targets:
        print("Working on predicting {}".format(col))
        scoring_dict[col] = dict()
        Y = df[col].values.tolist()
        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        for model in models:
            scoring_dict, best_model = globals()[model](X, Y, lookup_dict, scoring_dict, col, kfold, seed)

        #y_hat = best_model.predict(X_test)
        #print(classification_report(y_test, y_hat, best_model.classes_, digits=3))
        """
        for metric in metrics:
            if metric == 'accuracy':
                score = metric_mapping[metric](y_test, y_hat)
            else:
                score = metric_mapping[metric](y_test, y_hat, average='macro')
            scoring_dict[col][model][metric] = "{0:.3f}".format(score)
        """

    scoring_output = os.path.join(scoring_dir, config_text, "-".join(f for f in feature_methods))
    write_evaluations(scoring_dict, scoring_output, output_name)
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


################## Classification Methods ##################

def log_regression(X, Y, lookup_dict, scoring_dict, col, kfold, seed):
    #X = np.array(X, dtype=np.float32)
    #Y = np.array(Y)
    #Y = Y.astype(int)
    num_classes = list(set(Y))
    model = "log_regression"
    scoring_dict[col][model] = dict()
    if len(num_classes) == 2:
        print("Logistic Regression")
        class_type = 'ovr'
    elif len(num_classes) > 1:
        print("Multinomial Logistic Regression")
        class_type = 'multinomial'

    log_model = LogisticRegression(class_weight=None, random_state=seed, 
                                   solver='sag', multi_class=class_type, verbose=0,
                                   tol=1e-4, max_iter=1000)
    choose_model = GridSearchCV(log_model, cv=kfold, iid=True, 
                                param_grid={"class_weight": [None, 'balanced'],
                                            "C": np.arange(0.1, 1.0, 0.1)})
    choose_model.fit(X, Y)
    best_model = choose_model.best_estimator_
    best_params = choose_model.best_params_
    scoring_dict[col][model]['params'] = best_params

    # New CV code
    log_model = LogisticRegression(class_weight=best_params["class_weight"], random_state = seed,
                                   solver='sag', multi_class=class_type, verbose=0,
                                   tol=1e-4, max_iter=1000, C=best_params['C'])
    num_validations = 10
    metrics = ['f1_macro', 'accuracy']
    result_metrics = {item: list() for item in metrics}
    for met in metrics:
        for i in range(num_validations):
            run_scores = cross_val_score(log_model, X, Y, 
                                         cv=10,
                                         scoring=met)
            result_metrics[met] += run_scores.tolist()
    scoring_dict[col][model]["scores"] = result_metrics
    coefficients = best_model.coef_
    for i in range(len(coefficients)):
        class_name = best_model.classes_[i]
        coef = coefficients[i]
        coef_dict = {j: val for j, val in enumerate(coef)}
        word_coefs = {lookup_dict[j]: val for j, val in coef_dict.items()}
        abs_val_coefs = {word: abs(val) for word, val in word_coefs.items()}
        top_features = sorted(abs_val_coefs.items(), key=operator.itemgetter(1), reverse=True)[:100]
        real_weights = [[word, word_coefs[word]] for word, _ in top_features]
        scoring_dict[col][model][str(class_name) + '_top_features'] = real_weights
    return scoring_dict, best_model



def svm(X, Y, lookup_dict, scoring_dict, col, kfold, seed):
    model = "svm"
    scoring_dict[col][model] = dict()
    svm_model = LinearSVC(dual=False, random_state=seed)
    choose_model = GridSearchCV(svm_model, cv=kfold, iid=True, scoring='accuracy',
                                param_grid={'class_weight': [None, 'balanced'],
                                            'C': np.arange(0.1, 1.0, 0.1)})
    choose_model.fit(X, Y)
    best_model = choose_model.best_estimator_
    scoring_dict[col][model]['params'] = choose_model.best_params_

    coefficients = best_model.coef_
    for i in range(len(coefficients)):
        class_name = best_model.classes_[i]
        coef = coefficients[i]
        coef_dict = {j: val for j, val in enumerate(coef)}
        word_coefs = {lookup_dict[j]: val for j, val in coef_dict.items()}
        abs_val_coefs = {word: abs(val) for word, val in word_coefs.items()}
        top_features = sorted(abs_val_coefs.items(), key=operator.itemgetter(1), reverse=True)[:100]
        real_weights = [[word, word_coefs[word]] for word, _ in top_features]
        scoring_dict[col][model][str(class_name) + '_top_features'] = real_weights
    return scoring_dict, best_model

def write_evaluations(scoring_dict, scoring_output, output_name):

    if not os.path.exists(scoring_output):
        os.makedirs(scoring_output)
    scoring_output = os.path.join(scoring_output, output_name + ".json")
    with open(scoring_output, 'w') as fo:
        json.dump(scoring_dict, fo, indent=4)
