import pandas as pd
from sklearn import svm
from dataset import DatasetLoader
from typing import Any, Dict, List, Tuple
import json
import time
import re

def calculate_accuracy(y_pred: List[Any], y_true: List[Any], percentage: bool = True) -> Tuple[int, float]:
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true: correct += 1
    
    ratio = correct / len(y_true)
    
    return correct, round(ratio * 100, 4) if percentage else ratio

def format_json_key(kernel_name: str, params: Dict[str, Any]) -> str:
    param_list = []
    # Format parameters into a string for the key
    for k, v in params.items():
        if isinstance(v, str):
            v_str = v.replace("'", "").replace('"', "")
        else:
            v_str = str(v)
        param_list.append(f"{k}-{v_str}")
    
    param_str = "_".join(param_list)
    
    key = f"svm.svc_{kernel_name}"
    if param_str:
        key += f"_{param_str}"
    return key

def train_and_evaluate_svc(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    fold_index: int, 
    kernel: str, 
    svc_params: Dict[str, Any]
) -> Tuple[float, List[Any], List[Any], float]:
    
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1].tolist() 
    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1].tolist()
    
    start_time = time.time()
    
    model = svm.SVC(
        kernel=kernel, 
        C=1.0, 
        random_state=42, 
        **svc_params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).tolist()
    
    end_time = time.time()
    fold_time = end_time - start_time

    correct_count, accuracy_percentage = calculate_accuracy(y_pred=y_pred, y_true=y_test, percentage=True)
    accuracy_ratio = correct_count / len(y_test)
    
    # Print console output
    param_str = ", ".join(f"{k}={v}" for k, v in svc_params.items())
    label = f"{kernel.upper()}: ({param_str})" if param_str else f"{kernel.upper()}"
    print(f"   Fold {fold_index} [{label}]: Accuracy = {accuracy_percentage:.4f}% ({correct_count}/{len(y_test)}) | Time: {fold_time:.4f}s")
    
    return accuracy_ratio, y_test, y_pred, fold_time

if __name__ == "__main__":
        
    kernel_configs = [
        ("linear", {}),                                  # Linear: ⟨x,x′⟩
        ("poly", {"degree": 3, "coef0": 0.0}),           # Polynomial: (γ⟨x,x′⟩+r)^d
        ("rbf", {"gamma": 'scale'}),                     # Rbf: exp(−γ‖x−x′‖^2)
        ("sigmoid", {"coef0": 0.0, "gamma": 'scale'}),   # Sigmoid: tanh(γ⟨x,x′⟩+r)
    ]

    dataset_names = ["credit-a", "pen-based"]
    all_kernel_results = {} 

    for dataset in dataset_names:
        print(f"\n================ DATASET: {dataset} ================")
        
        dataset_loader = DatasetLoader(dataset_name=dataset)
        dataset_loader.load()
        
        full_dataset_results = {}
        
        # Loop through each kernel configuration
        for kernel_name, params in kernel_configs:
            print(f"\n--- Evaluating Kernel: {kernel_name.upper()} (Params: {params}) ---")
            
            fold_results_for_kernel = {} # Stores results for all folds for the current kernel
            fold_accuracy_ratios = []
            
            # Loop through each cross-validation fold
            for i, (df_train, df_test) in enumerate(dataset_loader):
                accuracy_ratio, y_true, y_pred, fold_time = train_and_evaluate_svc(
                    df_train, 
                    df_test, 
                    i + 1, # Fold index
                    kernel=kernel_name,
                    svc_params=params
                )
                
                fold_accuracy_ratios.append(accuracy_ratio)
                
                fold_results_for_kernel[str(i)] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "fold_accuracy": round(accuracy_ratio * 100, 4), 
                    "fold_time": round(fold_time, 6)
                }
            
            if fold_accuracy_ratios:
                # Calculate and print the average accuracy for the kernel
                avg_accuracy_ratio = sum(fold_accuracy_ratios) / len(fold_accuracy_ratios)
                avg_accuracy_percentage = round(avg_accuracy_ratio * 100, 4)
                print(f"\n--- Average {kernel_name.upper()} Accuracy: {avg_accuracy_percentage:.4f}% ---")
                
                # Format the key and store the detailed fold results in the final structure
                json_key = format_json_key(kernel_name, params)
                full_dataset_results[json_key] = fold_results_for_kernel
                
                # Store average accuracy for the final console summary
                all_kernel_results[dataset] = all_kernel_results.get(dataset, {})
                all_kernel_results[dataset][kernel_name] = avg_accuracy_percentage

        with open(f"resultssvm_{dataset}.json", "w+") as f:
            json_str = json.dumps(full_dataset_results, indent=3)
            json_str = re.sub(r"\[\s+([\d.,\s]+?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", json_str)
            f.write(json_str)
            
    print("\n================ FINAL RESULTS ================")
    for dataset, results in all_kernel_results.items():
        print(f"Dataset: {dataset}")
        for kernel, acc in results.items():
            print(f"  {kernel.upper()} Average Accuracy: {acc:.4f}%")

