import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import json

def load_results(file_paths: list[str]):
    data = {}
    
    
    for path in file_paths:
        df_name = path.split(".")[0].split("_")[-1]
        with open(path, "r") as f:
            file_content = json.load(f)
        
        for model_name, results in file_content.items():
            # if not model_name.startswith("euc"): continue
            if model_name not in data:
                data[model_name] = {}
            # data[model_name][df_name] = results["time"]
            # print(results)
            for fold, fold_res in results.items():
                if type(fold_res) != dict: continue

                col_name = f"{df_name}_{fold}"
                accuracy = fold_res["fold_accuracy"] / 100
                data[model_name][col_name] = accuracy
    
            
    df = pd.DataFrame.from_dict(data)
    print(df)
    
    return df

# Helper functions for performing the statistical tests
def generate_scores(method, method_args, data, labels):
    pairwise_scores = method(data, **method_args) # Matrix for all pairwise comaprisons
    pairwise_scores = pairwise_scores.set_axis(labels, axis='columns') # Label the cols
    pairwise_scores = pairwise_scores.set_axis(labels, axis='rows') # Label the rows, note: same label as pairwise combinations
    return pairwise_scores

def plot(scores):
    # Pretty plot of significance
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'square': True,
                    'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

    sp.sign_plot(scores, **heatmap_args)
    plt.show()
    



def compute_nemenyi(df: pd.DataFrame, alpha: float):
    cds = {
        0.01: 30.1642512887,
        0.05: 27.5194099696,
        0.1: 26.2094608262
    }
    # CD_0.01 = 4.557802422 * sqrt(72*(72+1) / (6 * 20)) = 30,1642512887
    # CD_0.05 = 4.158168297 * sqrt(72*(72+1) / (6 * 20)) = 27,5194099696
    # CD_0.1 = 3.960235674 * sqrt(72*(72+1) / (6 * 20)) = 26,2094608262
    
    # We expect as many rows as datasets and as many columns as algorithms (accuracies)
    algorithms = df.columns
    num_algorithms = len(algorithms)
    ranks = df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values() # Algorithm -> average rank
    print(avg_ranks)
    
    results = np.zeros(shape=(num_algorithms, num_algorithms))
    
    for i, alg_1 in enumerate(algorithms):
        for j, alg_2 in enumerate(algorithms):
            alg_1_rank = avg_ranks[alg_1]
            alg_2_rank = avg_ranks[alg_2]
            diff = np.abs(alg_2_rank - alg_1_rank)
            significant = 1 if diff >= cds[alpha] else 0
            
            results[i, j] = significant

    return results
    
if __name__ == "__main__":
    
    res_paths = [
        "results_credit-a.json",
        "results_pen-based.json",
        # "results_vowel.json",
        # "results_grid.json",
    ]
    
    df = load_results(res_paths)
    
    data = np.asarray(df)
    results = compute_nemenyi(df, 0.1)
    plt.imshow(results)
    plt.colorbar()
    # plt.show()
    # exit()
    # To be safe, ensure this matches what was expected
    num_datasets, num_methods = data.shape
    print("Methods:", num_methods, "Datasets:", num_datasets)
    
    print(df)
    # ranks = df.rank(axis=1, ascending=False)
    
    # print(ranks)
    # print(ranks.iloc[[0]])
    # for col, val in dict(sorted(ranks.iloc[0].items(), key=lambda x: x[1])).items():
    #     print(col, val)
    
    # print("Average ranks:\n", avg_ranks)
    data = np.asarray(df)
    
    num_datasets, num_algorithms = data.shape
    print("Algorithms:", num_algorithms, "Datasets:", num_datasets)
    # FRIEDMAN TEST
    alpha = 0.05

    stat, p = stats.friedmanchisquare(*data)
    print(f"p-value: {p}")

    print(stat, p)
    reject = p <= alpha
    print("Should we reject H0 (i.e. is there a difference in the classifiers) at the", (1-alpha)*100, "% confidence level?", reject)


    # NEMENY TEST (post-hoc)

    
    nemenyi_scores = generate_scores(sp.posthoc_nemenyi_friedman, {}, data, df.columns)
    print(np.min(nemenyi_scores))
    print(nemenyi_scores.iloc[9].values)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(nemenyi_scores, vmin=0, vmax=1)
    plt.colorbar()
    
    x, y = np.where(nemenyi_scores == np.min(nemenyi_scores))
    x = np.unique(x)
    y = np.unique(y)
    
    plt.xticks(ticks=np.arange(len(nemenyi_scores.columns)), labels=nemenyi_scores.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(nemenyi_scores.index)), labels=nemenyi_scores.index)


    plt.savefig("nemeny-pvalue_2.png", dpi=300)
    # plt.show()
    
    # TODO: find a reference that explains if it is possible to reject a null hypothesis in the friedman test (or big test) but do not reject any of the post-hoc.
    # TODO: find a reference that explains whether in the friedman test, it is required for the number of datasets to be larger than the number of models, and if so, by which margin