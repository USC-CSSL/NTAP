import pandas as pd

class CV_Results:
    def __init__(self, results):
        # assume results is one CV run (handle grid search elsewhere)
        res_by_name = dict()
        for r in results:
            for row in r:
                t = row["Target"]
                if t not in res_by_name:
                    res_by_name[t] = [row]
                else:
                    res_by_name[t].append(row)
        self.dfs = list()
        for target, cv_res in res_by_name.items():
            cv_df = pd.DataFrame(cv_res)
            cv_df.index.name = target
            cv_df.drop(columns=["Target"], inplace=True)
            means = [cv_df[col].mean() for col in cv_df.columns]
            cv_df.loc["Mean", :] = means
            self.dfs.append(cv_df)
    
    def summary(self):
        for df in self.dfs:
            print(df)
            print("-" * 100)
