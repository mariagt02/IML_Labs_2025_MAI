import os
import numpy as np
import pandas as pd
from scipy.io import arff

DATA_PATH = "data"
CLEAN_PATH = "clean_data"

dataset_1 = "credit-a"
dataset_2 = "pen-based"

"""
TODO:

Combine training + testing dataset, preprocess and split once again.

- Preprocessing: 
    Numerical Variables
    Categorical Variables
    
    Imputation -> KNN neighbors imputation (for categorical variables)
               -> Median (for numerical variables)
    
    34 rows with missing values in the traiing dataset
"""


def preprocess_dataset(df: pd.DataFrame):
    
    nan_rows = df.isna().any(axis=1)
    nan_cols = df.isna().any(axis=0)
    

def parse_dataset(dataset_name: str, src_path: str="data", dst_path: str="data_clean"):
    dataset_path = os.path.join(src_path, dataset_name)

    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        if not filename.endswith(".arff"): continue
        # print(filename)
        df, meta = arff.loadarff(filepath)
        print(meta.names())
        preprocess_dataset(pd.DataFrame(df, columns=meta.names()).replace({b"?": np.nan}))

        break



parse_dataset("credit-a")