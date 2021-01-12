from typing import Optional, Iterable
import optuna
import pandas as pd
import numpy as np

class SupervisedSummary:

    metric_list = ['f1', 'precision', 'recall', 'accuracy', 'r2', 'mse',
                   'rmse']

    def __init__(self, formula: str, task: str, params: dict, scoring_metric: str):

        self.formula = formula
        print(self.formula)
        self.task = task
        self.params = params
        self.scoring_metric = scoring_metric
        #self.model_info = model_info

    def load_sklearn_validator(self, gridcv_dict):

        best_idx = np.argmin(gridcv_dict[f'rank_test_{self.scoring_metric}'])

        self.best_score = gridcv_dict[f'mean_test_{self.scoring_metric}'][best_idx]
        self.best_params = gridcv_dict['params'][best_idx]

        self.scores = dict()

        for metric in self.metric_list:
            #if metric == self.scoring_metric:
                #continue
            metric_name = f'mean_test_{metric}'
            if metric_name in gridcv_dict:
                self.scores[metric] = gridcv_dict[metric_name][best_idx]

        for score, value in self.scores.items():
            print(f'{score:<10} {value:>10.3f}')

    def __repr__(self):

        return (f"Model trained to maximize {self.scoring_metric}\n"
                f"Achieved {self.best_score:.2f}")

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

