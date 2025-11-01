import pandas as pd
from sklearn import svm
from dataset import DatasetLoader
from typing import Any
import json
import time
import itertools
from utils import GlobalConfig, TerminalColor, calculate_accuracy, pretty_json_format
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from k_ibl import ReductionTechnique
from red_techniques import Reductor


def format_test_name(kernel_name: str, params: dict[str, Any], c: float, reduction: str) -> str:
    """
    Formats a test name for SVM experiments based on kernel, parameters, and C value.
    
    Args:
        kernel_name (str): The name of the SVM kernel (e.g., 'linear', 'rbf').
        params (dict[str, Any]): A dictionary of SVM parameters.
        c (float): The regularization parameter C.
    
    Returns:
        str: A formatted test name string.
    """
    param_list = []
    # Format parameters into a string for the key
    for k, v in params.items():
        if isinstance(v, str):
            v_str = v.replace("'", "").replace('"', "")
        else:
            v_str = str(v)
        param_list.append(f"{k}-{v_str}")
    
    param_str = "_".join(param_list)

    key = f"svm.svc_{kernel_name}_c_{c}"
    if param_str:
        key += f"_{param_str}"
        
    key += f"_reduction_{reduction}" if reduction != "None" else ""
    return key

def svmAlgorithm(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    kernel: str, 
    svc_params: dict[str, Any],
    c: float = 1.0
) -> tuple[float, list[Any], list[Any], float]:
    """
    Implements the SVM algorithm using scikit-learn's SVC.
    
    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The testing dataset.
        kernel (str): The SVM kernel to use (e.g., 'linear', 'rbf').
        svc_params (dict[str, Any]): Additional parameters for the SVC.
        c (float): The regularization parameter C.
    
    Returns:
        tuple[float, list[Any], list[Any], float]: A tuple containing:
            - accuracy_ratio (float): The accuracy as a ratio (0 to 1).
            - y_true (list[Any]): The true labels from the test set.
            - y_pred (list[Any]): The predicted labels from the SVM model.
            - fold_time (float): The time taken to train and test the model.
    """

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1].tolist() 
    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1].tolist()
    
    start_time = time.time()
    
    model = svm.SVC(
        kernel=kernel,  
        C=c,
        random_state=42, 
        **svc_params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).tolist()
    
    end_time = time.time()
    fold_time = end_time - start_time

    correct_count, _ = calculate_accuracy(y_pred=y_pred, y_true=y_test, percentage=True)
    accuracy_ratio = correct_count / len(y_test)
    return accuracy_ratio, y_test, y_pred, fold_time



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SVM Experiments")
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=GlobalConfig.DEFAULT_PREPROCESSED_DATASET_DIR,
        help="Directory where preprocessed datasets are stored."
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="List of dataset names to run experiments on. Use 'all' to run on all available datasets."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GlobalConfig.DEFAULT_SVM_RESULTS_PATH,
        help="Directory where output files will be saved."
    )
    
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="If set, existing output files will be overwritten. Otherwise, datasets for which results already exist will be skipped."
    )
    
    parser.add_argument(
        "-s", "--summary",
        action="store_true",
        help="If set, a summary of the results will be printed at the end."
    )

    parser.add_argument(
        "--reduction_technique",
        nargs="+",
        type=str,
        default=["None"],
        choices=ReductionTechnique.get_all_values() + ["all", "None"],
        help="List of reduction techniques to use. Use 'all' to include all available techniques."
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: number of CPU cores)."
    )

    parsed_args = parser.parse_args()
    
    if "all" in parsed_args.datasets and len(parsed_args.datasets) > 1:
        parser.error("If 'all' is specified in datasets, no other dataset names can be provided.")
    if "all" in parsed_args.reduction_technique and len(parsed_args.reduction_technique) > 1:
        parser.error("If 'all' is specified in reduction_technique, no other techniques can be provided.")
    
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    return parsed_args



def run_single_test(dataset: str, kernel_name: str, params: dict, c: float, dataset_dir: str, reduction: ReductionTechnique) -> tuple[str, str, dict]:
    dataset_loader = DatasetLoader(dataset_name=dataset, dataset_dir=dataset_dir)
    dataset_loader.load()
    
    test_name = format_test_name(kernel_name, params, c, reduction)
    
    fold_results_for_kernel = {}
    fold_accuracy_ratios = []
    
    experiment_start_time = time.time()
    for i, (df_train, df_test) in enumerate(dataset_loader):
        cols = df_train.columns.tolist()
        df_train = df_train.to_numpy()
        if reduction == ReductionTechnique.ALL_KNN:
            df_train = Reductor.ALLKNN.reduce(data=df_train, k=5, ivdm_metric=None, metric="euc")
        elif reduction == ReductionTechnique.ICF:
            df_train = Reductor.ICF.reduce(data=df_train, k=5)
        elif reduction == ReductionTechnique.MCNN:
            df_train = Reductor.MCNN.reduce(data=df_train)
        
        # Not the most intelligent approach (convert to numpy to reduce, then back to DataFrame), but it works for now.
        df_train = pd.DataFrame(df_train, columns=cols)
        
        accuracy_ratio, y_true, y_pred, fold_time = svmAlgorithm(
            df_train, 
            df_test,
            kernel=kernel_name,
            svc_params=params,
            c=c
        )
        fold_accuracy_ratios.append(accuracy_ratio)
        fold_results_for_kernel[str(i)] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "fold_accuracy": round(accuracy_ratio * 100, 4),
            "fold_time": round(fold_time, 6)
        }
    
    avg_accuracy = round(sum(fold_accuracy_ratios) / len(fold_accuracy_ratios) * 100, 4)
    fold_results_for_kernel["total_accuracy"] = avg_accuracy
    fold_results_for_kernel["total_time"] = time.time() - experiment_start_time
    
    return dataset, test_name, fold_results_for_kernel


if __name__ == "__main__":
    
    args = parse_args()
    
    if args.datasets == ["all"]:
        dataset_names = [dataset_name for dataset_name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, dataset_name))]
    else:
        dataset_names = args.datasets
    
    reduction = args.reduction_technique
    if reduction == ["all"]:
        reduction = ReductionTechnique.get_all_values() + ["None"]

    
    kernel_configs = [
        ("linear", {}),
        
        ("poly", {"degree": 1, "coef0": 0.0, "gamma": "scale"}),
        ("poly", {"degree": 1, "coef0": 0.0, "gamma": "auto"}),
        ("poly", {"degree": 3, "coef0": 0.0, "gamma": "scale"}),
        ("poly", {"degree": 3, "coef0": 0.0, "gamma": "auto"}),
        ("poly", {"degree": 5, "coef0": 0.0, "gamma": "scale"}),
        ("poly", {"degree": 5, "coef0": 0.0, "gamma": "auto"}),

        ("poly", {"degree": 1, "coef0": 1.0, "gamma": "scale"}),
        ("poly", {"degree": 1, "coef0": 1.0, "gamma": "auto"}),
        ("poly", {"degree": 3, "coef0": 1.0, "gamma": "scale"}),
        ("poly", {"degree": 3, "coef0": 1.0, "gamma": "auto"}),
        ("poly", {"degree": 5, "coef0": 1.0, "gamma": "scale"}),
        ("poly", {"degree": 5, "coef0": 1.0, "gamma": "auto"}),

        ("rbf", {"gamma": "scale"}),
        ("rbf", {"gamma": "auto"}),

        ("sigmoid", {"coef0": 0.0, "gamma": "scale"}),
        ("sigmoid", {"coef0": 0.0, "gamma": "auto"}),
        ("sigmoid", {"coef0": 1.0, "gamma": "scale"}),
        ("sigmoid", {"coef0": 1.0, "gamma": "auto"}),
    ]
    # kernel_configs = [
    #     ("rbf", {"gamma": "scale"}),
    # ] # Uncomment to test for the best SVM configuration
    
    Cs = [0.1, 1.0, 10.0]
    # Cs = [10.0] # Uncomment to test for the best SVM configuration
    param_combs = list(itertools.product(*[kernel_configs, Cs, reduction]))

    
    num_tests = len(param_combs) * len(dataset_names)
    
    all_kernel_results = {} 

    tasks = []
    for dataset in dataset_names:
        output_path = os.path.join(args.output_dir, f"results_{dataset}.json")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"{TerminalColor.colorize('Skipping', color='red', bold=True)} dataset {TerminalColor.colorize(dataset, color='yellow')}")
            continue

        for (kernel_name, params), c, reduction in param_combs:
            
            tasks.append((dataset, kernel_name, params, c, reduction))

    print(f"{TerminalColor.colorize('Running', color='green', bold=True)} {len(tasks)} tests in parallel using {TerminalColor.colorize(str(args.workers), color='yellow')} workers...")

    all_kernel_results = {}
    full_dataset_results = {d: {} for d in dataset_names}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_test, dataset, kernel, params, c, args.dataset_dir, reduction): (dataset, kernel)
            for (dataset, kernel, params, c, reduction) in tasks
        }

        for test_num, future in enumerate(as_completed(futures), start=1):
            try:
                dataset, test_name, fold_results = future.result()
                print(f"Test [{test_num} / {len(tasks)}]. Dataset: {TerminalColor.colorize(dataset, color='yellow')}. Hyperparameters: {TerminalColor.colorize(test_name, color='orange', bold=True)} done. Total accuracy: {TerminalColor.colorize(fold_results['total_accuracy'], color='green', bold=True)}%")
                full_dataset_results[dataset][test_name] = fold_results
            except Exception as e:
                print(f"{TerminalColor.colorize('Error', color='red', bold=True)} in one test: {e}")

    for dataset, results in full_dataset_results.items():
        if not results: continue
        
        output_path = os.path.join(args.output_dir, f"results_{dataset}.json")
        with open(output_path, "w+") as f:
            f.write(pretty_json_format(results))

    # if args.summary:
    #     for dataset, results in full_dataset_results.items():
    #         output_summary_path = os.path.join(args.output_dir, f"{dataset}_summary.txt")
    #         with open(output_summary_path, "w+") as f:
    #             f.write(f"Dataset: {dataset}\n")
    #             # Write the average accuracy of the dataset in the file