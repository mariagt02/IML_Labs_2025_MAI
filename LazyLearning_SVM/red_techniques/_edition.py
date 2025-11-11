from typing import Literal
from metrics import Metrics
import numpy as np

__all__ = ["AllKNN"]



class AllKNN:

    @staticmethod
    def reduce(data: np.ndarray, k: int, metric: Literal["euc", "cos", "ivdm"]="euc", ivdm_metric: Metrics.IVDM = None):
        
        # print(f'\nOriginal data shape: {data.shape}')

        keep_instance_flags = np.ones(data.shape[0], dtype=bool)

        if metric == "euc":
            metric_func = Metrics.Base.euclidean_dist
        elif metric == "cos":
            metric_func = Metrics.Base.cosine_dist
        elif metric == "ivdm":
            if ivdm_metric is not None:
                metric_func = ivdm_metric.compute
            else:
                # Iterate over the columns fo the data. If they are integer, then it is a discrete col. Otherwise, it is a continuous col
                discrete_cols = []
                continuous_cols = []
                for col in range(data.shape[1]-1):
                    col_element = data[0, col]
                    if isinstance(col_element, np.integer):
                        discrete_cols.append(col)
                    elif isinstance(col_element, np.floating):
                        continuous_cols.append(col)
                metric_func = Metrics.IVDM(data, discrete_cols, continuous_cols)
                metric_func = metric_func.compute
                
        else:
            raise ValueError(f"Unknown metric: {metric}")

        for i in range(data.shape[0]): # for each instance in data


            instance = data[i, :]
            label = data[i, -1]

            if metric == "ivdm":
                sorted_neighbours = metric_func(instance)[1:]
                # [1:] to skip the instance itself from the closest neighbors (nearest neighbor is itself, dist = 0)
            else:
                sorted_neighbours = metric_func(data, instance)[1:]

            for j in range(k): # for each neighbour of the instance
                
                # print('\nITERATION', j)

                i_nearest = sorted_neighbours[:j+1] # keep as many neighbours as the iteration j indicates

                vals, counts = np.unique(i_nearest[:, -1], return_counts=True)

                most_voted_label = vals[np.argmax(counts)]
                majority_votes = np.max(counts)
                
                total_votes = np.sum(counts)
                majority_fraction = majority_votes / total_votes

                if majority_fraction > 0.5: # només si és la majoria estricta (<50% of votes) --> per no tenir empats
                    has_strict_majority = True
                else:
                    has_strict_majority = False
                
                # if at some point the closest neighbours disagree (majoria != instance label), take the instance out
                if has_strict_majority and most_voted_label != label:
                    keep_instance_flags[i] = 0  # remove
                    break

        
        reduced_data = data[keep_instance_flags]
        # print(f'Reduced data shape: {reduced_data.shape}')
        return reduced_data

