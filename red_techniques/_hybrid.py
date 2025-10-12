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
        for i, x in enumerate(X):
            instance_repeated = np.tile(x, (X.shape[0], 1))
            euclidean_dist = np.linalg.norm(instance_repeated[:, :-1] - X[:, :-1], axis=1)
            sorted_idx = np.argsort(euclidean_dist)
            y_idx_sorted = y[sorted_idx]
            target_y = y[i]
            recheable_ = []
            for i, y_i in enumerate(y_idx_sorted):
                if y_i == target_y:
                    recheable_.append(sorted_idx[i])
                else:
                    break
            recheable_list.append(recheable_)
            
        print([len(elem) for elem in recheable_list])

        return [np.array(elem) for elem in recheable_list]
        # return recheable_list

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
        reachable = ICF._reachable(X, y)
        coverage = ICF._coverage(reachable)
        progress = False
        while not progress:
            X_r = X.copy()
            y_r = y.copy()
            reachable_r = deepcopy(reachable)
            coverage_r = deepcopy(coverage)
            
            
            for i, x in enumerate(X):
                if len(reachable[i]) > len(coverage[i]):
                    X_r = np.delete(X_r, i, axis=0)
                    y_r = np.delete(y_r, i, axis=0)

                    reachable_r.pop(i)
                    coverage_r.pop(i)
                    
                else:
                    progress = True
            X = X_r.copy()
            y = y_r.copy()
            reachable = deepcopy(reachable_r)
            coverage = deepcopy(coverage_r)
        return np.concatenate((X_r, y_r), axis = 1)

