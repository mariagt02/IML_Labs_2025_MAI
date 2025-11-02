import os
import numpy as np
import pandas as pd
from scipy.io import arff
import argparse
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

DEFAULT_INPUT_PATH = "data"
DEFAULT_OUTPUT_PATH = "preprocessed"

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
    
    34 rows with missing values in the trainig dataset
"""


'''
To check (Maria):
- Columna A15: segons el GitHub aquell és 'income' (ingressos). Ha valors molt alts (around 30k) i altres que són 0 --> després de normalitzar alguns valors queden molt petits 0'000... Té sentit, però no sé si perdem info
- Columna A14: segons el GitHub és el ZipCode. S'ha de tractar com una variable numèrica? No és una variable contínua com pes/edat/etc. És discreta. És numèrica però a la vegada categòrica (??)
'''
'''
To check (Natalia):
- Quan categoritzem les categoriques les de 2 es queden en int i les altres en float --> les pasem totes a int o a bool?
- class esta en categorical, seria mejor sacarla para no hacer ningun procesamiento a esa columna, tambien deberiamos cambiar la de is_train
'''

def impute_categorical(
    df: pd.DataFrame,
    categorical_columns: list,
    imp_strategy_categorical: str = "most_frequent"
):
    # Perform categorical data imputation
    if imp_strategy_categorical == "KNNImputer":
        # Encoding
        df_encoded = pd.DataFrame(index=df.index)
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df_notnull = df[col].dropna()
            le.fit(df_notnull) # Fit with no null values
            encoders[col] = le
            df_encoded[col] = df[col].map(lambda x: le.transform([x])[0] if pd.notna(x) else np.nan)
        # KNN
        imp_categorical = KNNImputer(n_neighbors=5)
        df_imputed_num = pd.DataFrame(imp_categorical.fit_transform(df_encoded), columns = categorical_columns, index=df.index)
        # Decoding
        for col in categorical_columns:
            le = encoders[col]
            df[col] = df_imputed_num[col].round().astype(int)
            df[col] = le.inverse_transform(df[col])
        
    elif imp_strategy_categorical == "most_frequent":
        imp_categorical = SimpleImputer(missing_values=np.nan, strategy=imp_strategy_categorical)
        for col in categorical_columns:
            df[col] = imp_categorical.fit_transform(df[[col]]).ravel()
    else:
        raise ValueError(f"Invalid imputation strategy for categorical data: {imp_strategy_categorical}")

    return df


def impute_numerical(
    df: pd.DataFrame,
    numerical_columns: list,
    imp_strategy_numerical: str = "median"
):
    imp_numerical = SimpleImputer(missing_values=np.nan, strategy=imp_strategy_numerical)
    for col in numerical_columns:
        df[col] = imp_numerical.fit_transform(df[[col]]).ravel()
    return df

def impute_nans(
    df: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list,
    imp_strategy_numerical: str = "median",
    imp_strategy_categorical: str = "KNNImputer",
    print_results: bool = True
):
    """
    Remove missing values using imputation
    
    Parameters:
    - df: Input DataFrame
    - numerical_columns: List of numerical column names to impute
    - categorical_columns: List of categorical column names to impute
    - imp_strategy_numerical: Imputation strategy for numerical data ('median', 'mean', etc.)
    - imp_strategy_categorical: Imputation strategy for categorical data ('most_frequent', KNNImputer)
    - print_results: print the before and after results or not
    """

    df_copy = df.copy()

    # Perform numerical data imputation
    if len(numerical_columns) > 0:
        df_copy = impute_numerical(df_copy, numerical_columns=numerical_columns, imp_strategy_numerical=imp_strategy_numerical)
    
    if len(categorical_columns) > 0:
        df_copy = impute_categorical(df_copy, categorical_columns=categorical_columns, imp_strategy_categorical=imp_strategy_categorical)

    # Find rows with NaN values
    nan_mask = df.isnull().any(axis=1)
    nan_rows = df[nan_mask]
    display_indices = []
    # Print results
    if print_results:
        print(f"\n=== {len(nan_rows)} ROWS WITH MISSING VALUES (BEFORE IMPUTATION) ===")
        if len(nan_rows) > 0:
            display_rows = nan_rows.head(5)
            print("Showing up to 5 rows with missing values:")
            print(display_rows)
            display_indices = display_rows.index.tolist()
        else:
            print("No nan values in numerical columns")
    
    if print_results and display_indices:
        print(f"\n=== SAME ROWS AFTER IMPUTATION ===")
        print(df_copy.loc[display_indices])
    
    return df_copy

def normalize_data(df: pd.DataFrame, numerical_columns: list, categorical_columns: list):
    df_out = df.copy()
    encoders = {}

    # Numerical columns normalization (min-max scaling)
    scaler = preprocessing.MinMaxScaler()
    df_out[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # One-hot encoding
    for col in categorical_columns:
        unique_vals = pd.Series(df[col].dropna().unique())

        if len(unique_vals) == 2:
            le = LabelEncoder()
            df_out[col] = le.fit_transform(
                df[col].astype(str)
            )
            encoders[col] = le

        elif len(unique_vals) > 2:
            ohe = OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore',
                dtype = np.int8
            )
            transformed = ohe.fit_transform(df[[col]].astype(str))
            
            new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            ohe_df = pd.DataFrame(
                transformed, columns=new_cols, index=df.index
            )

            df_out = pd.concat([df_out.drop(columns=[col]), ohe_df], axis=1)
            encoders[col] = ohe

    return df_out

def preprocess_dataset(
    df: pd.DataFrame,
    imp_strategy_numerical:str ="median",
    imp_strategy_categorical: str="KNNImputer",
    skip_cols: list[str] = ["is_train", "a17"],
    target_cols: list[str] = ["class", "a17"],
    print_results: bool=False
):
    # df = df.replace({b"?": np.nan}) FIXME: this is not needed, as it is already done when converting from arff to pd
  
    # Divide columns in numerical and categorical
    numerical_columns = [col for col in df.select_dtypes(include=["number"]).columns.tolist() if col not in skip_cols]
    categorical_columns = [col for col in df.select_dtypes(include=["object"]).columns.tolist() if col not in skip_cols]

    df_clean = impute_nans(
        df=df[[col for col in df.columns if col not in skip_cols]],
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        imp_strategy_numerical=imp_strategy_numerical,
        imp_strategy_categorical=imp_strategy_categorical,
        print_results=print_results
    )
    
    df_encoded = normalize_data(df=df_clean, numerical_columns=numerical_columns, categorical_columns=categorical_columns)
    # Add the columns that we have skipped to the final dataset
    
    for col in skip_cols:
        if col not in df.columns: continue
        df_encoded[col] = df[col]
    
    for col in target_cols:
        # Add the target column at the end of the dataframe (for clarity purposes)
        if col not in df.columns: continue
        df_encoded = df_encoded[[c for c in df_encoded.columns if c != col] + [col]]
 
    return df_encoded
    

def arff_to_pd(path: str, col_names: list[str]=None) -> pd.DataFrame:
    
    df, meta = arff.loadarff(path)
    
    if col_names:
        cols = meta.names()
        assert len(cols) == len(col_names)
    else:
        cols = col_names
    
    df = pd.DataFrame(df, columns=cols).replace({b"?": np.nan})
    
    
    # The `arff.loadarff` function loads string attributes as bytes. This function converts those byte columns to strings
    # for better handling during preprocessing.
    for col in df.select_dtypes([object]).columns:
        if isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode("utf-8")
    
    return df


def merge_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["is_train"] = True
    test_df["is_train"] = False
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    # Delete lines containing more than 3 NaN values (those with more than 3 all have 5).
    combined_df = combined_df[~(combined_df.isna().sum(axis=1) > 3)]

    return combined_df


def split_fold(df: pd.DataFrame, train_test_col: str="is_train") -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df[train_test_col]].drop(columns=[train_test_col])#.reset_index(drop=True)
    df_test = df[~df[train_test_col]].drop(columns=[train_test_col])#.reset_index(drop=True)
    return df_train, df_test


def store_preprocessed_df(df_train: pd.DataFrame, df_test: pd.DataFrame, fold: int, df_name: str, out_path: str) -> None:
    output_dir = os.path.join(out_path, df_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    train_file = os.path.join(output_dir, f"{df_name}.fold.{str(fold).zfill(6)}.train.csv")
    test_file = os.path.join(output_dir, f"{df_name}.fold.{str(fold).zfill(6)}.test.csv")

    for df, file_path in [(df_train, train_file), (df_test, test_file)]:
        if os.path.exists(file_path):
            print(f"Warning: {file_path} already exists.")
            choice = input("Press [Y/y] to replace. Anything else to skip: ").strip().lower()
            if choice == "y":
                df.to_csv(file_path, index=False)
                print(f"Replaced {file_path}.")
            else:
                print(f"Skipped {file_path}.")
        else:
            df.to_csv(file_path, index=False)
            print(f"Created {file_path}.")



def parse_dataset(dataset_name: str) -> list[pd.DataFrame]:
    
    train_files = {}
    test_files = {}
    folds_found = 0
    
    for file in os.listdir(dataset_name):
        # Read the names of all files to store them in a structured way
        if not file.endswith(".arff"): continue
        
        file_parts = file.split(".")
        fold_num = int(file_parts[2])
        
        if file_parts[3] == "test": test_files[fold_num] = os.path.join(dataset_name, file)
        if file_parts[3] == "train": train_files[fold_num] = os.path.join(dataset_name, file)
        
        folds_found = max(fold_num, folds_found)
    
    
    preprocessed_dfs = []
    for i in range(folds_found + 1):
        # Parse train and test files for each fold
        train_file = train_files[i]
        test_file = test_files[i]
        
        train_df = arff_to_pd(train_file)
        test_df = arff_to_pd(test_file)
        
        fold_df = merge_fold(train_df, test_df)
        
        preprocessed_dfs.append(fold_df)
    
    return preprocessed_dfs




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset preprocessor",
        description="A simple Python script for preprocessing datasets used in the Master's in Artificial Intelligence course: Introduction to Machine Learning"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Names of specific datasets to preprocess. If not provided, all datasets in the input folder will be processed."
    )
    parser.add_argument(
        "--input_path",
        default=DEFAULT_INPUT_PATH,
        help="Path to the folder containing the datasets. Default is 'data'."
    )
    parser.add_argument(
        "--output_path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path where preprocessed datasets will be saved. Default is 'preprocessed'."
    )
    
    args = parser.parse_args()
    
    datasets = args.datasets
    input_path = args.input_path
    output_path = args.output_path
    
    if not datasets:
        datasets = []
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            if os.path.isdir(file_path):
                datasets.append(file_path)
    else:
        datasets = [
            os.path.join(input_path, df_name) for df_name in datasets
        ]
            
    for df_name in datasets:
        # For every fold, obtain a single dataset with all the training and testing instances.
        dfs = parse_dataset(df_name)
        
        
        for i, df in enumerate(dfs):
            # Preprocess the dataset for each fold.
            # Yo aqui meteria el df sin la columna de class ni la de is_train
            df_preprocessed = preprocess_dataset(df)
            
            # Split the dataset back into training and testing
            df_train, df_test = split_fold(df_preprocessed)
            
            # Store each dataset in its corresponding file
            store_preprocessed_df(df_train, df_test, fold=i, df_name=df_name.split("/")[-1], out_path=output_path)
            
            # break # So far, simply perprocess the first fold