import os
import pandas as pd
import numpy as np

"""
Decided: Do message-level and user-level metrics separately (do both, don't make everything user-level)

To Do:
    - For each CV: 
        - Compute accuracy (k_hat != hat)
        - Compute MSE
    - Rank each message by average accuracy and MSE (get hard-to-classify and easy-to-classify)
"""

if __name__ == '__main__':
    param_path = os.environ['PARAMS']
    pred_path = os.environ['PRED_PATH']
    #source_path = os.environ['SOURCE_PATH']

    """
    source_df = pd.read_pickle(source_path)
    print(source_df)
    """

    for f in os.listdir(pred_path):
        print(f)
        if not f.endswith('pkl'):
            continue
        path_to_file = os.path.join(pred_path, f)
        pred_df = pd.read_pickle(path_to_file)
        
        for cv_fold, cv_df in pred_df.groupby(level=0):
            print(cv_fold)
            # accuracy
            num_correct = 0
            scores = list()
            for mess_id, row in cv_df.iterrows():
                scores.append(abs(row['y'] - row['y_hat']))
                if scores[-1] == 0:
                    num_correct += 1
            print(( 1. * num_correct) / len(cv_df))
            print(np.mean(np.array(scores)))
            
