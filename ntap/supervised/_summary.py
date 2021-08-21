import sys
from itertools import zip_longest
import time
from typing import Optional, Iterable
import optuna
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble

from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (gen_fmt, fmt_2,
                                               fmt_params, fmt_2cols)


class MethodInfo:
    method_list = {
        'least_squares': {
            'backend': 'sklearn.linear_model',
            'classifier': linear_model.LogisticRegression(),
            'regressor': linear_model.LinearRegression(),
            'params': {'C': 10.0 ** np.arange(-3, 3, 1)}
        },
        'svm-lin': {
            'backend': 'sklearn.svm',
            'classifier': svm.LinearSVC(max_iter=1000),
            'regressor': svm.LinearSVR(max_iter=1000),
            'params': {'C': 10.0 ** np.arange(-3, 3, 1)}
        },
        'svm': {
            'backend': 'sklearn.svm',
            'classifier': svm.SVC(),
            'regressor': svm.SVR(),
            'params': {'C': 10.0 ** np.arange(-3, 3, 1),
                       'kernel': ['poly', 'rbf', 'linear'],
                       'gamma': 10.0 ** np.arange(-2, 3)}
        },
        'tree-ensemble': {
            'backend': 'sklearn.ensemble',
            'classifier': ensemble.GradientBoostingClassifier(),
            'regressor': ensemble.GradientBoostingRegressor()
        },
        'naive-bayes': {
            'backend': 'sklearn',
            'classifier': None
        },
        'lstm': {
            'backend': 'pytorch',
            'classifier': None,
            'regressor': None
        },
        'finetune': {
            'backend': 'transformers',
            'classifier': None,
            'regressor': None}
   #'sequence': ['finetune']}
    }

    def __init__(self, method_desc, task, num_classes=None):

        self.method_desc = method_desc

        if method_desc not in self.method_list:
            #TODO: fuzzy str matching
            raise ValueError(f"{method_desc} not available. Options: "
                             f"{' '.join(list(self.method_list.keys()))}")
            return

        self.backend = self.method_list[method_desc]['backend']

        if task == 'classify':
            self.model = self.method_list[method_desc]['classifier']
            self.__check_classifier()
            if num_classes is None:
                raise ValueError("`num_classes` cannot be None when task "
                                 "is `classify`")
            else:
                self.num_classes = num_classes
        elif task == 'regress':
            self.model = self.method_list[method_desc]['regressor']
            self.__check_regressor()
        else:
            raise ValueError(f"Could not identify task parameter `{task}`")
        self.task = task

    def get_param_grid(self):
        grid = self.method_list[self.method_desc]['params']
        if self.task == 'classify':
            grid['class_weight'] = [None, 'balanced']
        return grid

    def __check_classifier(self):
        if self.model is None:
            raise NotImplementedError(f"{self.method_desc} has no classifier implemented")

    def __check_regressor(self):
        if self.model is None:
            raise NotImplementedError(f"{self.method_desc} has no regressor implemented")

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), compact=True)


class SupervisedSummary:

    metric_list = ['f1', 'precision', 'recall', 'accuracy', 'r2', 'mse',
                   'rmse']
    print_width = 82

    def __init__(self, formula: str, task: str, method_info: MethodInfo,
                 params: dict, scoring_metric: str, warnings:
                 Iterable[str] = list()):

        self.formula = formula
        self.task = task
        self.params = params
        self.scoring_metric = scoring_metric
        self.warnings = warnings
        self.method_info = method_info
        #self.header = f'{self.task.capitalize()}: {self.formula}'
        #self.header = f'{self.header:^{self.print_width}}'

        #self.description = (f'Prediction method(s) {self.method_info}')

        self.feature_models = list()

    def load_sklearn_validator(self, validator):

        self.validator = validator
        gridcv_dict, best_model = (validator.cv_results_,
                                   validator.best_estimator_)

        best_idx = np.argmin(gridcv_dict[f'rank_test_{self.scoring_metric}'])

        self.best_score = gridcv_dict[f'mean_test_{self.scoring_metric}'][best_idx]
        self.best_params = gridcv_dict['params'][best_idx]
        self.best_model = best_model

        self.scores = dict()

        for metric in self.metric_list:
            metric_name = f'mean_test_{metric}'
            if metric_name in gridcv_dict:
                self.scores[metric] = gridcv_dict[metric_name][best_idx]

        self.scoring_tbl = []

        for score, value in self.scores.items():
            output = f'{score:<10} {value:>10.3f}'
            if score == self.scoring_metric:
                output += '*'
            self.scoring_tbl.append(output)

    def add_feature_models(self, feature_model):
        self.feature_models = feature_model

    def print(self, file=sys.stdout):

        #file.write(self.header)
        #file.write('\n' + '=' * self.print_width + '\n')

        time_now = time.localtime()
        time_of_day = [time.strftime("%H:%M:%S", time_now)]
        date = time.strftime("%a, %d %b %Y", time_now)

        gen_left = [('Model:',
                     [self.method_info.model.__class__.__name__]),
                    ('Backend:', [self.method_info.backend]),
                    ('Date:', [date])]
        gen_stubs_left, gen_data_left = zip_longest(*gen_left)

        gen_title = f'{self.task}:{self.formula}'
        gen_title = f'{gen_title:^{self.print_width}}'
        gen_table_left = SimpleTable(gen_data_left,
                                     headers=['Model Info'],
                                     stubs=gen_stubs_left,
                                     title=gen_title)

        #gen_right = [('Validation:', [self.validator.__class__.__name__, 
                                     #f"{self.validator.n_splits_}-fold"]),
                     #('Filler1:', ['some stuff', 'fuck']),
                     #('Filler2:', ['some stuff', 'shit'])]

        gen_right = [('Validation:', [self.validator.__class__.__name__]),
                     ('Filler1:', ['some stuff']),
                     ('Filler2:', ['some stuff'])]
        gen_stubs_right, gen_data_right = zip_longest(*gen_right)
        print(gen_data_right)
        gen_table_right = SimpleTable(gen_data_right,
                                      headers=['Fitting Info'],
                                      stubs=gen_stubs_right,
                                      title=gen_title)
        print(gen_table_right.as_text())
        gen_table_left.extend_right(gen_table_right)
        #print(gen_table_left.as_text())
        has_feat = False

        if has_feat:
            feat_title = "Feature Info"
            feature_keys = list(self.feature_models.feature_models.keys())
            feature_display = ' '.join(feature_keys)

            feat_left = [('Model:', [feature_display]),
                         ('Trained on:', ['Some corpus summary'])]
            feat_stubs_left, feat_data_left = zip_longest(*feat_left)

            feat_tbl_left = SimpleTable(feat_data_left,
                                        headers=None,
                                        stubs=feat_stubs_left,
                                        title=feat_title)
            print(feat_tbl_left)
            #gen_table_left.extend(feat_tbl_left)

        #file.write('\n'.join(self.scoring_tbl))

        if len(self.warnings) > 0 and False:
            file.write('\n')
            file.write(f'At least {len(self.warnings)} warnings were '
                       'raised during fitting. Run `see_warnings()` '
                       'to access.')

    def see_warnings(self, top_n=10, file=sys.stdout):
        if len(self.warnings) > top_n:
            # only print top_n warnings
            file.write('\n'.join([str(w.message) for w in self.warnings[:top_n]]))
            file.write(f'\nTruncated ({len(self.warnings) - top_n} '
                       'more)\n')
        else:
            file.write('\n'.join([str(w.message) for w in self.warnings]))
            file.write('\n')

#class _ModelGrid:

class ValidatedModel:
    def __init__(self, 
                 study: optuna.Study, 
                 feature_matrix: Optional[np.ndarray] = None,
                 target_vector: Optional[np.ndarray] = None,
                 result_type: str = 'cross_val',
                 prediction_task: str = 'classification'):

        self.study = study
        self.result_type = result_type

        best_params = self.study.best_params.items()
        best_params = ['{}: {:.2f}'.format(k, v) if isinstance(v, float)
                       else '{}: {}'.format(k, v) for k, v in best_params]

    def summary(self):
        print("Run {} trials\n"
              "Best {} ({:.3f}) with params:\n"
              "{}".format(len(study.trials), scoring_metric, study.best_value, 
                          '\n'.join(best_params)))

    def get_all_runs(self) -> pd.DataFrame:
        return self.study.trials_dataframe()

    def print_confusion_matrix(self):
        if self.task != 'classify':
            raise RuntimeError("Confusion matrix unavailable for regression problems")

