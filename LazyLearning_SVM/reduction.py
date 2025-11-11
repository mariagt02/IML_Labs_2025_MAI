from red_techniques import Reductor
from dataset import DatasetLoader, DatasetVisualizer
from k_ibl import ReductionTechnique, IBLHyperParameters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import argparse
from utils import GlobalConfig, TerminalColor
import os
import itertools


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualization of the reduction techniques effects for the given datasets."
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
        "--output_dir",
        type=str,
        default=GlobalConfig.DEFAULT_REDUCTION_VISUALIZATIONS_PATH,
        help="Directory where output files will be saved."
    )

    reduction_group = parser.add_mutually_exclusive_group(required=True)
    reduction_group.add_argument(
        "--mcnn",
        action="store_true",
        help="Use MCNN reduction technique."
    )
    reduction_group.add_argument(
        "--icf",
        action="store_true",
        help="Use ICF reduction technique (requires --k)."
    )
    reduction_group.add_argument(
        "--allknn",
        action="store_true",
        help="Use All-KNN reduction technique (requires --k and --metric)."
    )
    
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of neighbors for ICF reduction."
    )
    
    parser.add_argument(
        "--metric",
        nargs="+",
        type=str,
        default=["all"],
        choices=IBLHyperParameters.get_all_sim_metrics() + ["all"],
        help="List of similarity metrics to use with AllKNN reduction. Use 'all' to include all available metrics."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pca",
        action="store_true",
        help="Use PCA for dimensionality reduction."
    )
    group.add_argument(
        "--tsne",
        action="store_true",
        help="Use t-SNE for dimensionality reduction."
    )
    
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="If set, saves the original dataset visualization before reduction."
    )
    
    parsed_args = parser.parse_args()
    
    if "all" in parsed_args.datasets and len(parsed_args.datasets) > 1:
        parser.error("If 'all' is specified in datasets, no other dataset names can be provided.")
    if "all" in parsed_args.metric and len(parsed_args.metric) > 1:
        parser.error("If 'all' is specified in metric, no other metrics can be provided.")

    
    if parsed_args.icf and parsed_args.k is None:
        parser.error("Argument --icf requires --k to be specified.")
    if parsed_args.allknn:
        if parsed_args.k is None:
            parser.error("Argument --allknn requires --k to be specified.")
        if parsed_args.metric is None:
            parser.error("Argument --allknn requires --metric to be specified.")
    
    
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    return parsed_args


def reduce_df(df: np.ndarray, reduction_technique: ReductionTechnique, k: int, metric: str) -> np.ndarray:
    if reduction_technique == ReductionTechnique.MCNN:
        reduced_df = Reductor.MCNN.reduce(df)
    elif reduction_technique == ReductionTechnique.ICF:
        reduced_df = Reductor.ICF.reduce(df, k=k)
    elif reduction_technique == ReductionTechnique.ALL_KNN:
        reduced_df = Reductor.ALLKNN.reduce(df, k=k, metric=metric)

    
    return reduced_df



if __name__ == "__main__":
    args = parse_args()
    
    if args.datasets == ["all"]:
        dataset_names = [dataset_name for dataset_name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, dataset_name))]
    else:
        dataset_names = args.datasets

    reduction = [ReductionTechnique.MCNN if args.mcnn else (ReductionTechnique.ICF if args.icf else ReductionTechnique.ALL_KNN)]
    ks = [args.k] if (args.icf or args.allknn) else [None]
    metrics = args.metric if args.metric != ["all"] else IBLHyperParameters.get_all_sim_metrics()
    
    metrics = metrics if args.allknn else [None]

    combs = list(itertools.product(dataset_names, reduction, ks, metrics))
    

    use_pca = args.pca
    use_tsne = args.tsne
    
    
    for dataset, reduction, k, metric in combs:
        print(f"Processing dataset: {TerminalColor.colorize(dataset, 'yellow')} with reduction: {TerminalColor.colorize(reduction.value, 'orange')}. {'PCA' if use_pca else 't-SNE'} for visualization. K={k}. Metrics: {metric}")
        name = f"{dataset}_{reduction.name}_{k}_{metric}" if k is not None and metric is not None else (f"{dataset}_{reduction.name}_{k}" if k is not None else f"{dataset}_{reduction.name}")
        name = f"{name}_pca" if use_pca else f"{name}_tsne"
        
        dataset_loader = DatasetLoader(
            dataset_name=dataset,
            dataset_dir=args.dataset_dir
        )
        
        dataset_loader.load()
        
        train_df, test_df = dataset_loader[0]
        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()
        complete_df = train_df # Numpy ndarray with the train and test dataset
        
        
        visualizer = DatasetVisualizer(dataset_name=dataset_loader.dataset_name)
        X_reduced, y, fig = visualizer.visualize_df(num_dims=2, df=train_df, dim_reduction="pca" if use_pca else "tsne", show=False)
        
        if args.save_original:
            output_path = os.path.join(args.output_dir, f"{name}_original.png")
            plt.savefig(output_path, dpi=300)
            print(f"Saved original dataset visualization to {TerminalColor.colorize(output_path, 'green')}")
        
        train_reduced = reduce_df(train_df, reduction_technique=reduction, k=k, metric=metric)
        print(f"Original dataset size: {len(train_df)}. Reduced dataset size: {len(train_reduced)}")
        
        complete_df_rows = {tuple(row): i for i, row in enumerate(complete_df)}
        indices = [complete_df_rows[tuple(row)] for row in train_reduced if tuple(row) in complete_df_rows]
        
        X_reduced_sub = X_reduced[indices, :]
        y_sub = y[indices]
        
        fig = visualizer.visualize_reduced_df(X_reduced_sub, y_sub, dataset, reduction.value, num_dims=2, show=False, dim_reduction="pca" if use_pca else "tsne")
        output_path = os.path.join(args.output_dir, f"{name}_reduced.png")
        plt.savefig(output_path, dpi=300)
        print(f"Saved reduced dataset visualization to {TerminalColor.colorize(output_path, 'green')}")
        