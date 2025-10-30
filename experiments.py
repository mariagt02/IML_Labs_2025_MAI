# This script runs experiments using the K-IBL algorithm on specified datasets with various hyperparameter combinations.

from k_ibl import KIBLearner, IBLHyperParameters, ReductionTechnique
import json
import itertools
from sklearn.neighbors import KNeighborsClassifier
from dataset import DatasetLoader
from utils import TerminalColor
import time
import argparse
from utils import GlobalConfig
import os


def calculate_accuracy(y_pred: list[int], y_true: list[int], percentage: bool = True) -> tuple[int, float]:
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true: correct += 1
    
    ratio = correct / len(y_true)
    
    return correct, round(ratio * 100, 4) if percentage else ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-IBL experiments on specified datasets with various hyperparameter combinations."
    )
    
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
        "--ks",
        nargs="+",
        type=int,
        default=[3, 5, 7],
        help="List of k values for the K-IBL algorithm."
    )
    
    parser.add_argument(
        "--retention",
        nargs="+",
        type=str,
        default=["all"],
        choices=IBLHyperParameters.get_all_retention_policies() + ["all"],
        help="List of retention policies to use. Use 'all' to include all available policies."
    )
    
    parser.add_argument(
        "--similarity_metrics",
        nargs="+",
        type=str,
        default=["all"],
        choices=IBLHyperParameters.get_all_sim_metrics() + ["all"],
        help="List of similarity metrics to use. Use 'all' to include all available metrics."
    )

    parser.add_argument(
        "--voting_schemes",
        nargs="+",
        type=str,
        default=["all"],
        choices=IBLHyperParameters.get_all_voting_schemes() + ["all"],
        help="List of voting schemes to use. Use 'all' to include all available schemes."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GlobalConfig.DEFAULT_RESULTS_PATH,
        help="Directory where output files will be saved."
    )
    
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="If set, existing output files will be overwritten. Otherwise, datasets for which results already exist will be skipped."
    )

    parsed_args = parser.parse_args()
    
    if "all" in parsed_args.datasets and len(parsed_args.datasets) > 1:
        parser.error("If 'all' is specified in datasets, no other dataset names can be provided.")
    if "all" in parsed_args.retention and len(parsed_args.retention) > 1:
        parser.error("If 'all' is specified in retention, no other retention policies can be provided.")
    if "all" in parsed_args.similarity_metrics and len(parsed_args.similarity_metrics) > 1:
        parser.error("If 'all' is specified in similarity_metrics, no other metrics can be provided.")
    if "all" in parsed_args.voting_schemes and len(parsed_args.voting_schemes) > 1:
        parser.error("If 'all' is specified in voting_schemes, no other schemes can be provided.")

    os.makedirs(parsed_args.output_dir, exist_ok=True)

    return parsed_args


if __name__ == "__main__":
    
    args = parse_args()
    
    # Dataset selection
    if args.datasets == ["all"]:
        dataset_names = [dataset_name for dataset_name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, dataset_name))]
    else:
        dataset_names = args.datasets
    
    
    k_values = args.ks
    
    
    
    if args.similarity_metrics == ["all"] and args.voting_schemes == ["all"] and args.retention == ["all"]:
        hyperparameters = IBLHyperParameters.get_all_values() + [k_values]
    else:
        excluded_sim_metrics = [metric for metric in IBLHyperParameters.get_all_sim_metrics() if metric not in args.similarity_metrics] if args.similarity_metrics != ["all"] else []
        excluded_voting_schemes = [scheme for scheme in IBLHyperParameters.get_all_voting_schemes() if scheme not in args.voting_schemes] if args.voting_schemes != ["all"] else []
        excluded_retention = [policy for policy in IBLHyperParameters.get_all_retention_policies() if policy not in args.retention] if args.retention != ["all"] else []
        excluded_hyperparameters = excluded_sim_metrics + excluded_voting_schemes + excluded_retention
        hyperparameters = IBLHyperParameters.get_all_values(exclude=excluded_hyperparameters) + [k_values]
    
    # Use of pre-implemented knn to double-check our results
    
    # df_train = pd.read_csv(f"preprocessed/pen-based/pen-based.fold.000000.train.csv")
    # df_test = pd.read_csv(f"preprocessed/pen-based/pen-based.fold.000000.test.csv")
    # knn = KNeighborsClassifier(n_neighbors=7, metric="cosine")
    # knn.fit(df_train[df_train.columns[:-1]], df_train[df_train.columns[-1]])
    # y_pred = knn.predict(df_test[df_test.columns[:-1]])
    # y_true = df_test[df_test.columns[-1]]
    # accuracy = np.sum(y_pred == y_true) / len(y_true)
    # print(f"KNN accuracy: {round(accuracy * 100, 4)}%")
    
    # exit()
    hyperparameters_combinations = list(itertools.product(*hyperparameters))
    num_tests = len(hyperparameters_combinations) * len(dataset_names)
    
    
    test_num = 1
    for dataset in dataset_names:
        output_path = os.path.join(args.output_dir, f"results_{dataset}.json")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"{TerminalColor.colorize('Skipping', color='red', bold=True)} dataset {TerminalColor.colorize(dataset, color='yellow')} as results file already exists at {TerminalColor.colorize(output_path, color='green')}. Use --overwrite to force re-computation.")
            continue
        
        
        dataset_loader = DatasetLoader(dataset_name=dataset, dataset_dir=args.dataset_dir)
        dataset_loader.load()
        
        results = {}
        for metric, voting, retention, k in hyperparameters_combinations:
            test_name = f"{metric}_{voting}_{k}_{retention}"
            print(f"Test [{test_num} / {num_tests}]. Dataset: {TerminalColor.colorize(dataset_loader.dataset_name, color='yellow')}. Hyperparameters: {TerminalColor.colorize(test_name, color='orange', bold=True)}")
            ibl_learner = KIBLearner(
                sim_metric=metric,
                k=k,
                voting=voting,
                retention=retention
            )
            total_accuracy = 0
            results[test_name] = {}

            experiment_start_time = time.time()
            for i, (df_train, df_test) in enumerate(dataset_loader):
                fold_start_time = time.time()
                y_pred = ibl_learner.KIBLAlgorithm(df_train, df_test)
                fold_total_time = time.time() - fold_start_time
                y_true = df_test[df_test.columns[-1]]

                results[test_name][i] = {}

                results[test_name][i]["y_true"] = y_true.to_list()
                results[test_name][i]["y_pred"] = y_pred
                
                fold_correct, fold_accuracy = calculate_accuracy(y_pred, y_true.to_list())
                
                results[test_name][i]["fold_accuracy"] = fold_accuracy
                results[test_name][i]["fold_time"] = fold_total_time
                total_accuracy += fold_correct

            total_accuracy /= (len(y_pred) * dataset_loader.num_folds)            
            
            print(f"\tTotal accuracy: {TerminalColor.colorize(fold_accuracy, color='green', bold=True)}%")
            results[test_name]["total_accuracy"] = round(total_accuracy * 100, 4)
            results[test_name]["time"] = time.time() - experiment_start_time
        
            test_num += 1
        
        with open(output_path, "w+") as f:
            json.dump(results, f)
    