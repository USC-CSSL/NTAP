from typing import Optional, Iterable
import optuna
import pandas as pd
import numpy as np

class SupervisedSummary:

    def __init__(self, results, task, params, scoring_metric, model_info):
        self.results = results
        self.task = task
        self.params = params
        self.scoring_metric = scoring_metric
        self.model_info = model_info

        self.best_idx = np.argmin(results['rank_test_score'])
        self.best_score = results['mean_test_score'][self.best_idx]

    def __repr__(self):

        return (f"{self.model_info.backend} model trained "
                f"to maximize {self.scoring_metric}\n"
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

