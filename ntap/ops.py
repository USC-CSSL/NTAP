from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from sklearn.metrics import r2_score, cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .supervised import TextClassifier, TextRegressor

class ValidatedModel:
    def __init(self):
        pass

class ValidatedClassifier(ValidatedModel):

    def __init__(self, model, **kwargs):
        pass

    def cross_val(model, data, metrics=["f1", "accuracy", "precision", "recall"]):
        stats = list()
        y, y_hat = labels, predictions
        card = num_classes
        for y, y_hat in zip(predictions, labels):
            stat = {"Target": target}
            for m in metrics:
                if m == 'accuracy':
                    stat[m] = accuracy_score(y, y_hat)
                avg = 'binary' if card == 2 else 'macro'
                if m == 'precision':
                    stat[m] = precision_score(y, y_hat, average=avg)
                if m == 'recall':
                    stat[m] = recall_score(y, y_hat, average=avg)
                if m == 'f1':
                    stat[m] = f1_score(y, y_hat, average=avg)
                if m == 'kappa':
                    stat[m] = cohen_kappa_score(y, y_hat)
            stats.append(stat)
        return stats

    def cross_val_summary(model, data=None):
        # assume results is one CV run (handle grid search elsewhere)

        # TODO: get results from model
        res_by_name = dict()
        for r in results:
            for row in r:
                t = row["Target"]
                if t not in res_by_name:
                    res_by_name[t] = [row]
                else:
                    res_by_name[t].append(row)
        dfs = list()
        for target, cv_res in res_by_name.items():
            cv_df = pd.DataFrame(cv_res)
            cv_df.index.name = target
            cv_df.drop(columns=["Target"], inplace=True)
            means = [cv_df[col].mean() for col in cv_df.columns]
            std_devs = [cv_df[col].std() for col in cv_df.columns]
            cv_df.loc["Mean", :] = means
            cv_df.loc["Std",:] = std_devs
            dfs.append(cv_df)


class ValidatedRegressor(ValidatedModel):

    def __init__(self, model, folds, **kwargs):
        pass


def cross_validate(model, data, num_folds=10, random_seed=0):

    folds = KFold(n_splits=num_folds,
                  shuffle=True,
                  random_state=random_seed)

    # TODO: run CV with model, data, and folds

    if isinstance(model, TextClassifier):
        cv_obj = ValidatedClassifier(model, folds=folds, seed=random_seed)
    elif isinstance(model, TextRegressor):
        cv_obj = ValidatedRegressor(model, folds=folds, seed=random_seed)
    return cv_obj

def train(model, data):
    """Trains a model given data"""
    return


def predict(model, data):
    """ Given trained model (with formula specified), predicts labels """
    return

