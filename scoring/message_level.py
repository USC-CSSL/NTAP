import os
import pandas as pd
import numpy as np

"""
To Do:
    - For each CV: 
        - Compute accuracy (k_hat == k)
        - Compute MSE
    - Rank each message by average accuracy and MSE (get hard-to-classify and easy-to-classify)
"""

if __name__ == '__main__':
    param_path = os.environ['PARAMS']
    pred_path = os.environ['PRED_DIR']
    source_path = os.environ['SOURCE_DIR']

    for dir_ in os.listdir(pred_path):
        print(dir_)
        
        predictions = pd.read_pickle(os.path.join(pred_path, dir_, "predictions.pkl"))
        features = pd.read_pickle(os.path.join(pred_path, dir_, "features.pkl"))

        cv_metrics = list()
        for cv_fold, cv_df in predictions.groupby(level=0):
            # accuracy
            num_correct = 0
            scores = list()
            for mess_id, row in cv_df.iterrows():
                if row['y'] == row['y_hat']:
                    num_correct += 1
            cv_metrics.append(( 1. * num_correct) / len(cv_df))
        print(np.mean(cv_metrics))
            
