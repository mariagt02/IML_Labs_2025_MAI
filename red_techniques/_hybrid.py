import numpy as np
from metrics import Metrics
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
__all__ = ["ICF"]


class ICF:
    
    # @staticmethod
    # def wilson_editing(X, y):
    #     y_pred = []
    #     for _, instance in X.iterrows():
    #         instance = instance.to_numpy()
    #         nearest_outputs = ICF.return_nn(Metrics.Base.euclidean_dist(X, instance))
    #         output = ICF.modified_plurality(nearest_outputs)
    #         y_pred.append(output)
    #     mask = np.where(y_pred != y)
    #     X_r = np.delete(X, mask, axis=0)
    #     y_r = np.delete(y, mask, axis=0)
    #     return X_r, y_r
    
    @staticmethod
    def _wilson_editing(X, y, k):
        knn = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean').fit(X, y)
        y_pred = knn.predict(X)
        mask = np.where(y_pred != y)
        X_r = np.delete(X, mask, axis=0)
        y_r = np.delete(y, mask, axis=0)
        return X_r, y_r

    @staticmethod
    def _reachable(X, y):
        recheable_list = []
        X = np.array(X)
        y = np.array(y)
        for j, x in enumerate(X):
            instance_repeated = np.tile(x, (X.shape[0], 1))
            euclidean_dist = np.linalg.norm(instance_repeated[:, :-1] - X[:, :-1], axis=1)
            sorted_idx = np.argsort(euclidean_dist)
            y_idx_sorted = y[sorted_idx]
            target_y = y[j]
            recheable_ = []
            for i, y_i in enumerate(y_idx_sorted):
                if y_i == target_y:
                        if sorted_idx[i]!=j:
                            recheable_.append(sorted_idx[i])
                else:
                    break
            recheable_list.append(recheable_)

        return [np.array(elem) for elem in recheable_list]

    @staticmethod
    def _coverage(reachable):
        coverage_array = [[] for i in range(len(reachable))]
        for i, reachable_ in enumerate(reachable):
            for reach in reachable_:
                if i not in coverage_array[reach]:
                    coverage_array[reach].append(i)
        return [np.array(elem) for elem in coverage_array]

    @staticmethod
    def reduce(data, k):
        X=data[:,:-1]
        y=data[:,-1]
        X, y = ICF._wilson_editing(X, y, k)
        X = [np.array(elem) for elem in X]
        y = [np.array(elem) for elem in y]
        progress = True
        while progress:
            X_r = X.copy()
            y_r = y.copy()
            reachable = ICF._reachable(X, y)
            coverage = ICF._coverage(reachable)
            progress = False
            count_ = 0
            for i, x in enumerate(X):
                if len(reachable[i]) > len(coverage[i]):
                    X_r =[array for array in X_r if not np.array_equal(array, x)]
                    y_r.pop(i-count_)
                    count_ = count_+1
                    progress = True
            X = X_r.copy()
            y = y_r.copy()
        return np.column_stack((X_r, y_r))

