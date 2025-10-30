from k_ibl import KIBLearner, IBLHyperParameters, ReductionTechnique
import json
import itertools
import os
from sklearn.neighbors import KNeighborsClassifier
from dataset import DatasetLoader
from utils import TerminalColor, GlobalConfig
import time


def calculate_accuracy(y_pred: list[int], y_true: list[int], percentage: bool = True) -> tuple[int, float]:
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true: correct += 1
    
    ratio = correct / len(y_true)
    
    return correct, round(ratio * 100, 4) if percentage else ratio


if __name__ == "__main__":
    
    k = 7
    metric = 'euc'
    voting = 'bc'
    retention = 'dd'
    dataset_names = [
        "credit-a",
        # "pen-based"
    ]
    weighting = [None, "relief", "SFS"]
    
    num_tests = len(dataset_names)*len(weighting)
    test_num = 0
    for dataset in dataset_names:
        dataset_loader = DatasetLoader(dataset_name=dataset)
        dataset_loader.load()
        
        results = {}
        for weighting_method in weighting:
            test_name = f"{metric}_{voting}_{k}_{retention}_{weighting_method}"
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
                y_pred = ibl_learner.KIBLAlgorithm(df_train, df_test, weighted=weighting_method)
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
        
        
        with open(os.path.join(GlobalConfig.DEFAULT_WEIGHTED_RESULTS_PATH, f"results_{dataset}.json"), "w+") as f:
            json.dump(results, f)
    
