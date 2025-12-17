import os
import pandas as pd

class ResultTable:
    def __init__(self):
        self.rows = []

    def add(self, ratio, set_name, metrics_dict):
        self.rows.append({
            "ratio": ratio,
            "set": set_name,
            **metrics_dict
        })

    def to_csv(self, path):
        df = pd.DataFrame(self.rows)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def to_df(self):
        return pd.DataFrame(self.rows)
