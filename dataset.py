import os
import re
import pandas as pd
import string
from utils import TerminalColor, GlobalConfig
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class DatasetLoader:
    def __init__(
        self,
        dataset_name: str,
        filename_template="{dataset}.fold.{fold:06d}.{type}.{ext}",
        dataset_dir: str = "preprocessed",
        verbose: bool = False
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.filename_template = filename_template
        self.filename_regex = self._build_regex_from_template(filename_template)
        self.filename_pattern = re.compile(self.filename_regex)
        self.filename_map = {}
        self.num_folds = 0
        self.verbose = verbose

        self.__dataset_path = os.path.join(self.dataset_dir, self.dataset_name)
        if not os.path.isdir(self.__dataset_path):
            self._print(f"{TerminalColor().ERROR} dataset path {TerminalColor.colorize(self.__dataset_path, 'yellow')} does not exist")
            exit(-1)

    def _build_regex_from_template(self, template: str) -> str:
        """
        Convert a filename template into a regex pattern with named groups.

        This method dynamically parses placeholders (e.g. {name}, {fold:06d}) and
        assigns reasonable regex patterns based on the placeholder name or format spec.
        """
        formatter = string.Formatter()
        regex_parts = []

        for literal_text, field_name, format_spec, _ in formatter.parse(template):
            # Escape literal text
            if literal_text:
                regex_parts.append(re.escape(literal_text))

            # If this part is a literal, skip it.
            if not field_name:
                continue
            
            pattern = self._get_pattern_for_field(field_name, format_spec)
            regex_parts.append(pattern)
        
        # Combine and anchor regex
        regex = "".join(regex_parts)
        return f"^{regex}$"


    def _get_pattern_for_field(self, name: str, fmt: str | None) -> str:
        """
        Returns an appropriate regex group for a placeholder name and optional format.
        """
        if fmt and "d" in fmt: 
            width_match = re.search(r"0?(\d+)d", fmt)
            if width_match:
                width = int(width_match.group(1))
                return fr"(?P<{name}>\d{{{width}}})"
            return fr"(?P<{name}>\d+)"

        defaults = {
            "dataset": r"(?P<dataset>[a-zA-Z0-9_-]+)",
            "fold": r"(?P<fold>\d+)",
            "type": r"(?P<type>train|test)",
            "ext": r"(?P<ext>[a-z0-9]+)",
        }

        return defaults.get(name, fr"(?P<{name}>[a-zA-Z0-9_-]+)")

    def get_fold(self, fold_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the train and test data for a given fold number.

        Args:
            fold_num (int): The fold number to load.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple (train_df, test_df)
        """
        if self.num_folds == 0:
            self._print(f"{TerminalColor().ERROR} trying to load dataset that has not been loaded yet. Please, call .load() first.")
            exit(-1)

        if fold_num not in self.filename_map:
            self._print(f"{TerminalColor().ERROR} fold {TerminalColor.colorize(str(fold_num), 'yellow')} not found in loaded dataset.")
            exit(-1)

        fold_files = self.filename_map[fold_num]
        train_path = fold_files["train"]
        test_path = fold_files["test"]

        if not train_path or not os.path.isfile(train_path):
            self._print(f"{TerminalColor().ERROR} missing {TerminalColor.colorize('train', 'red')} file for fold {fold_num}")
            exit(-1)

        if not test_path or not os.path.isfile(test_path):
            self._print(f"{TerminalColor().ERROR} missing {TerminalColor.colorize('test', 'red')} file for fold {fold_num}")
            exit(-1)

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        except Exception as e:
            self._print(f"{TerminalColor().ERROR} failed to read CSV files for fold {fold_num}: {TerminalColor.colorize(str(e), 'red')}")
            exit(-1)

        return train_df, test_df


    
    def load(self):
        """
        Scans the dataset directory and builds a map of available folds.
        Each valid file (matching the filename pattern) is added to self.filename_map
        in the format:
            {
                fold_number: {
                    'train': '/path/to/file',
                    'test': '/path/to/file'
                },
                ...
            }
        """
        if not os.path.isdir(self.__dataset_path):
            self._print(f"{TerminalColor().ERROR} dataset path {TerminalColor.colorize(self.__dataset_path, 'yellow')} does not exist")
            exit(-1)

        self._print(f"Loading dataset files from {TerminalColor.colorize(self.__dataset_path, color='yellow')}")

        for filename in os.listdir(self.__dataset_path):
            file_path = os.path.join(self.__dataset_path, filename)

            if os.path.isdir(file_path):
                self._print(f"Skipping directory {TerminalColor.colorize(file_path, color='yellow')}...")
                continue

            match = self.filename_pattern.match(filename)
            if not match:
                self._print(
                    f"Skipping file {TerminalColor.colorize(filename, color='yellow')} "
                    f"as it does not follow the provided format "
                    f"{TerminalColor.colorize(self.filename_template, color='grey')}"
                )
                continue

            info = match.groupdict()
            fold = int(info["fold"])
            df_type = info["type"]

            # Initialize entry for this fold
            if fold not in self.filename_map:
                self.filename_map[fold] = {}

            self.filename_map[fold][df_type] = file_path

        
        # Count number of folds found
        self.num_folds = len(self.filename_map)

        if self.num_folds == 0:
            self._print(f"{TerminalColor().ERROR} No valid dataset folds found in {self.__dataset_path}")
            exit(-1)
        else:
            self._print(f"{TerminalColor().SUCCESS} loaded {self.num_folds} folds successfully.")

    def _print(self, msg) -> None:
        if self.verbose:
            print(msg)
    
    
    def __iter__(self):
        """
        Allows to iterate over the folds in ascending order of fold number
        """
        if self.num_folds == 0:
            self._print(f"{TerminalColor().ERROR} trying to load dataset that has not been loaded yet. Please, call .load() first.")
            exit(-1)
        
        for fold_num in sorted(self.filename_map.keys()):
            yield self.get_fold(fold_num)
            
            
    def __getitem__(self, index: int):
        """
        Allows to access the DatasetLoader in an index-based way.
        """
        return self.get_fold(index)
        


class DatasetVisualizer:
    def __init__(self,
        dataset_name: str,
        filename_template="{dataset}.fold.{fold:06d}.{type}.{ext}",
        dataset_dir: str = "preprocessed"
    ):
        self.df_loader = DatasetLoader(
            dataset_name=dataset_name,
            filename_template=filename_template,
            dataset_dir=dataset_dir,
            verbose=True
        )
        self.df_loader.load()
        
        self.__allowed_dims = [2, 3]
        self.AllowedDims = Literal[tuple(self.__allowed_dims)]
        
        self.custom_colors = [
            "#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF",
            "#00FFFF", "#FF8000", "#8000FF", "#008000", "#FF0080", "#000000"
        ]
    
    
    def visualize_df(self, num_dims: "DatasetVisualizer.AllowedDims", df: np.ndarray = None, dim_reduction: Literal["pca", "tsne"] = "tsne", show: bool = False) -> tuple[np.ndarray, np.ndarray, plt.Figure]:
        if num_dims not in self.__allowed_dims:
            raise ValueError(f"Received {num_dims}. Only allowed {self.__allowed_dims}")
        
        if df is None:
            train_df, test_df = self.df_loader.get_fold(1)
            df = pd.concat([train_df, test_df], axis=0)
            X = df[df.columns[:-1]]
            y = df[df.columns[-1]].to_numpy()
        else:
            X = df[:, :-1]
            y = df[:, -1]
        
        classes = np.unique(y)
        n_classes = len(classes)
        cmap = ListedColormap(self.custom_colors[:n_classes])
        
        reducer = TSNE(n_components=num_dims, random_state=42) if dim_reduction == "tsne" else PCA(n_components=num_dims)
        X_reduced = reducer.fit_transform(X)
        
        fig = self._plot(X_reduced, y, classes, cmap, num_dims, dim_reduction, title_prefix=f"{self.df_loader.dataset_name} original")
        if show:
            plt.show()
        return X_reduced, y, fig


    def visualize_reduced_df(self, X_reduced: np.ndarray, y: np.ndarray, dataset_name: str, reduction_name: str, num_dims: int = 2, show: bool = False, dim_reduction: Literal["pca", "tsne"] = "tsne") -> plt.Figure:
        if num_dims not in self.__allowed_dims:
            raise ValueError(f"Received {num_dims}. Only allowed {self.__allowed_dims}")
        
        classes = np.unique(y)
        n_classes = len(classes)
        cmap = ListedColormap(self.custom_colors[:n_classes])
        
        fig = self._plot(X_reduced, y, classes, cmap, num_dims, dim_reduction=dim_reduction, title_prefix=f"{dataset_name} reduced ({reduction_name})")
        if show:
            plt.show()
        return fig


    def _plot(self, X_reduced: np.ndarray, y: np.ndarray, classes: np.ndarray, cmap: ListedColormap, num_dims: int, dim_reduction: str = "", title_prefix: str = "") -> plt.Figure:
        fig = plt.figure(figsize=(10, 5))
        
        plt.rcParams.update({
            "text.usetex": True,
        })
        
        if num_dims == 2:
            for i, cls in enumerate(classes):
                idx = np.where(y == cls)
                plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], color=cmap(i), edgecolors="black", linewidth=0.2, label=int(float(str(cls))))
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.title(f"{dim_reduction.upper()} visualization of {title_prefix}", fontsize=14, fontweight="bold")
            plt.legend(title="Class")
        else:
            ax = plt.axes(projection="3d")
            for i, cls in enumerate(classes):
                idx = np.where(y == cls)
                ax.scatter3D(X_reduced[idx, 0].flatten(), X_reduced[idx, 1].flatten(), X_reduced[idx, 2].flatten(), color=cmap(i), label=int(float(str(cls))), edgecolor="black", linewidth=0.2)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.title(f"{dim_reduction.upper()} visualization of {title_prefix}", fontsize=14, fontweight="bold")
            plt.legend(title="Class")

        plt.tight_layout()
        return fig
        


if __name__ == "__main__":
    
    for df in ["credit-a", "pen-based", "vowel", "grid"]:            
        df_visualizer = DatasetVisualizer(
            dataset_name=df
        )
        
        _, _, fig = df_visualizer.visualize_df(num_dims=2, dim_reduction="pca", show=False)
        plt.savefig(os.path.join(GlobalConfig.DEFAULT_REDUCTION_VISUALIZATIONS_PATH, f"{df_visualizer.df_loader.dataset_name}_original_pca.png"), dpi=300)
        
        _, _, fig = df_visualizer.visualize_df(num_dims=2, dim_reduction="tsne", show=False)
        plt.savefig(os.path.join(GlobalConfig.DEFAULT_REDUCTION_VISUALIZATIONS_PATH, f"{df_visualizer.df_loader.dataset_name}_original_tsne.png"), dpi=300)
