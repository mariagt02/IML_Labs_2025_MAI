import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
import matplotlib.pyplot as plt
import argparse
import os
from utils import GlobalConfig, TerminalColor
from matplotlib.patches import Patch

plt.rcParams.update({
    "text.usetex": True,
})

class FriedmanTest:
    """
    Class to perform Friedman test and Nemenyi post-hoc test on model results.
    Attributes:
        file_paths (list[str]): List of file paths containing model results in JSON format.
        alpha (float): Significance level for the tests.
        df (pd.DataFrame): DataFrame to hold the loaded results.
        stat (float): Friedman test statistic.
        p_value (float): p-value from the Friedman test.
        nemenyi_p_matrix (np.ndarray): p-value matrix from Nemenyi post-hoc test.
        significance_matrix (np.ndarray): Binary matrix indicating significant differences.
        output_path (str): Path to save output results.
        _cmap_name (str): Colormap name for plotting.
    """
    def __init__(self, file_paths: list[str], alpha: float = 0.05, output_path: str = None, cmap: str = "jet"):
        """
        Initialize the FriedmanTest with file paths and significance level.
        """
        self.file_paths = file_paths
        self.alpha = alpha
        self.output_path = output_path
        self.df = None
        self.stat = None
        self.p_value = None
        self.nemenyi_p_matrix = None
        self.significance_matrix = None
        
        self._cmap_name = cmap

    def _load_results(self) -> None:
        """
        Load results from JSON files and organize them into a DataFrame.
        The structure of the JSON file is expected to be:
        {
            "model_name": {
                "fold_1": {"fold_accuracy": value, ...},
                "fold_2": {"fold_accuracy": value, ...},
                ...
            },
            ...
        }
        """
        data = {}

        for path in self.file_paths:
            df_name = path.split(".")[0].split("_")[-1]
            with open(path, "r") as f:
                file_content = json.load(f)

            for model_name, results in file_content.items():
                if model_name not in data:
                    data[model_name] = {}

                for fold, fold_res in results.items():
                    if not isinstance(fold_res, dict):
                        continue
                    col_name = f"{df_name}_{fold}"
                    accuracy = fold_res["fold_accuracy"] / 100
                    data[model_name][col_name] = accuracy

        self.df = pd.DataFrame.from_dict(data)

    def _compute_nemenyi_posthoc(self) -> None:
        """
        Compute the Nemenyi post-hoc test p-value matrix and significance matrix.
        Demšar, J. (2006). States that "The performance of two classifiers is significantly different
        if the corresponding average ranks differ by at least the critical difference", where
        the critical difference is:
                            CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
        where k is the number of algorithms, N is the number of datasets.
        q_alpha is obtained from the studentized range (sr) distribution divided by sqrt(2).
        The rejection of the null hypothesis is done if
                            |R_i - R_j| > sr / sqrt(2) * sqrt(k * (k + 1) / (6 * N))
        If we isolate sr, then
                            sr < |R_i - R_j| / sqrt(k * (k + 1) / (6 * N)) * sqrt(2)
        which gives us the p-value for the Nemenyi test.    
        """
        
        num_datasets, num_algorithms = self.df.shape
        algorithms = self.df.columns
        ranks = self.df.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean(axis=0).sort_values()

        results = np.zeros(shape=(num_algorithms, num_algorithms))
        for i, alg_1 in enumerate(algorithms):
            for j, alg_2 in enumerate(algorithms):
                diff = abs(avg_ranks[alg_2] - avg_ranks[alg_1])
                q_stat = diff / np.sqrt(num_algorithms * (num_algorithms + 1) / (6.0 * num_datasets))
                q_stat *= np.sqrt(2.0)
                results[i, j] = studentized_range.sf(q_stat, num_algorithms, np.inf)

        self.nemenyi_p_matrix = results
        self.significance_matrix = (results <= self.alpha).astype(int)

    def run(self):
        """
        Run the Friedman test and Nemenyi post-hoc test (as long as the null hypothesis for the Friedman test is rejected).
        """
        self._load_results()
        data = [self.df[col].values for col in self.df.columns]
        self.stat, self.p_value = stats.friedmanchisquare(*data)

        if self.p_value <= self.alpha:
            self._compute_nemenyi_posthoc()
        else:
            self.nemenyi_p_matrix = None
            self.significance_matrix = None

    def summary(self) -> dict:
        """
        Print a summary of the Friedman test and Nemenyi post-hoc test results into a text file.
        """
        out_file_path = os.path.join(self.output_path, f"summary_{self.alpha}.txt")
        f = open(out_file_path, "w")

        f.write(f"Friedman statistic: {self.stat}, p-value: {self.p_value}\n")
        
        reject_null = self.p_value <= self.alpha
        f.write(f"Can we reject the null hypothesis (all algorithms are equivalent) at alpha={self.alpha}? {'Yes' if reject_null else 'No'}\n")

        if reject_null:
            f.write(f"Null hypothesis has been rejected. This means that there is a significant difference between at least two algorithms.\n")
            f.write(f"To find out which algorithms differ significantly, we perform the Nemenyi post-hoc test.\n")
            
            num_significant = np.sum(self.significance_matrix) // 2
            f.write(f"Number of significant pairwise differences found: {num_significant}\n")

            # Print the name of the algorithms that differ significantly
            algorithms = self.df.columns
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    if self.significance_matrix[i, j] == 1:
                        f.write(f"\tSignificant difference between {algorithms[i]} and {algorithms[j]} (p-value: {self.nemenyi_p_matrix[i, j]:.4f})\n")

        f.close()
        print(f"Summary written to {TerminalColor.colorize(out_file_path, color='green')}")

    def plot_nemenyi_matrix(self):
        """
        Plot the Nemenyi p-value matrix as a heatmap.
        """
        if self.nemenyi_p_matrix is None:
            print("Nemenyi post-hoc test was not performed as the null hypothesis was not rejected.")
            return

        fig = plt.figure(figsize=(7, 7))
        plt.imshow(self.nemenyi_p_matrix, vmin=0, vmax=1, cmap=self._cmap_name, interpolation="nearest")
        plt.colorbar()
        plt.title("Nemenyi Post-hoc p-value Matrix")
        
        # Only plot ticks for those algorithms for which the significance value is 1
        significant_algorithms = np.where(self.significance_matrix == 1)[0]
        plt.xticks(ticks=significant_algorithms, labels=self.df.columns[significant_algorithms], rotation=90, fontsize=6)
        plt.yticks(ticks=significant_algorithms, labels=self.df.columns[significant_algorithms], fontsize=6)

        out_file_path = os.path.join(self.output_path, f"nemenyi_p_value_matrix_{self.alpha}.png")
        plt.savefig(out_file_path, dpi=300, bbox_inches="tight")
        
        print(f"Nemenyi p-value matrix plot saved to {TerminalColor.colorize(out_file_path, color='green')}")

    def plot_significance_matrix(self):
        """
        Plot the significance matrix as a heatmap.
        """
        if self.significance_matrix is None:
            print("Nemenyi post-hoc test was not performed as the null hypothesis was not rejected.")
            return

        fig = plt.figure(figsize=(7, 7))
        plt.imshow(self.significance_matrix, cmap=self._cmap_name, interpolation="nearest")
        # Show a legend with only two colors: 0 and 1
        cmap = plt.get_cmap(self._cmap_name)
        legend_elements = [
            Patch(facecolor=cmap(0.0), edgecolor='k', label='0'),
            Patch(facecolor=cmap(1.0), edgecolor='k', label='1'),
        ]
        plt.legend(handles=legend_elements, title='Significance', loc='upper right')
        plt.title("Nemenyi Post-hoc Significance Matrix")

        # Only plot ticks for those algorithms for which the significance value is 1
        significant_algorithms = np.where(self.significance_matrix == 1)[0]
        plt.xticks(ticks=significant_algorithms, labels=self.df.columns[significant_algorithms], rotation=90, fontsize=6)
        plt.yticks(ticks=significant_algorithms, labels=self.df.columns[significant_algorithms], fontsize=6)

        out_file_path = os.path.join(self.output_path, f"nemenyi_significance_matrix_{self.alpha}.png")
        plt.savefig(out_file_path, dpi=300, bbox_inches="tight")

        print(f"Nemenyi significance matrix plot saved to {TerminalColor.colorize(out_file_path, color='green')}")

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(
        prog="Friedman and Nemenyi Tests",
        description="Perform Friedman and Nemenyi tests on model results."
    )
    args.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Significance level for the tests."
    )

    # Add an argument to specify the base directory where the results files are stored
    args.add_argument(
        "--base_dir",
        type=str,
        default=GlobalConfig.DEFAULT_RESULTS_PATH,
        help="Base directory where the results files are stored."
    )

    # Add an argument to specify the datasets to include. Defaults to "all", which will make the script use all json files in the base directory.
    args.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="Datasets to include in the analysis. Use 'all' to include all datasets in the base directory."
    )
    
    # Mandatory argument to specify the name of the test that is being run
    args.add_argument(
        "--test_name",
        type=str,
        required=True,
        help="Name of the test being run"
    )
    
    # Optional argument to specify the output path for the results
    args.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output results"
    )
    

    parsed_args = args.parse_args()

    if "all" in parsed_args.datasets and len(parsed_args.datasets) > 1:
        args.error("If you specify 'all', no other dataset names can be provided.")

    
    # Build the list of result file paths based on the provided datasets
    res_paths = []
    if parsed_args.datasets == ["all"]:
        for file_name in os.listdir(parsed_args.base_dir):
            if file_name.endswith(".json"):
                res_paths.append(os.path.join(parsed_args.base_dir, file_name))
    else:
        for dataset in parsed_args.datasets:
            file_path = os.path.join(parsed_args.base_dir, f"results_{dataset}.json")
            if os.path.isfile(file_path):
                res_paths.append(file_path)
            else:
                args.error(f"Results file for dataset '{dataset}' not found at path: {file_path}")
    
    
    # If there is no output path specified, create one based on the test name
    if parsed_args.output_path is None:
        parsed_args.output_path = os.path.join(GlobalConfig.DEFAULT_STATS_OUTPUT_PATH, f"{parsed_args.test_name}")
    
    os.makedirs(parsed_args.output_path, exist_ok=True)

    test = FriedmanTest(res_paths, alpha=0.1, output_path=parsed_args.output_path)
    test.run()
    test.summary()
    test.plot_nemenyi_matrix()
    test.plot_significance_matrix()
