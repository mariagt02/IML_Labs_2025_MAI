from k_ibl import KIBLearner, IBLHyperParameters, ReductionTechnique
import json
import itertools
from sklearn.neighbors import KNeighborsClassifier
from dataset import DatasetLoader
from utils import TerminalColor
import time


def calculate_accuracy(y_pred: list[int], y_true: list[int], percentage: bool = True) -> tuple[int, float]:
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true: correct += 1
    
    ratio = correct / len(y_true)
    
    return correct, round(ratio * 100, 4) if percentage else ratio


if __name__ == "__main__":
    
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
    k_values = [3, 5, 7]
    
    hyperparameters = IBLHyperParameters.get_all_values() + [k_values]    
    hyperparameters_combinations = list(itertools.product(*hyperparameters))
    
    dataset_names = [
        "credit-a",
        "pen-based"
    ]
    num_tests = len(hyperparameters_combinations) * len(dataset_names)
    
    
    test_num = 0
    for dataset in dataset_names:
        dataset_loader = DatasetLoader(dataset_name=dataset)
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
                y_pred = ibl_learner.kIBLAlgorithm(df_train, df_test)
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
        
        
        # with open(f"results_{dataset}.json", "w+") as f:
        #     json.dump(results, f)
    