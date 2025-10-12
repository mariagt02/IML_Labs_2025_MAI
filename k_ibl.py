import pandas as pd
import numpy as np
from typing import Literal
import math
from collections import Counter
import json
import itertools
from sklearn.neighbors import KNeighborsClassifier
from metrics import Metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class KIBLearner:
    def __init__(self,
        sim_metric: Literal["euc", "cos", "heom", "ivdm", "gwhsm"],
        k: int,
        voting: Literal["mp", "bc"], # modified plurality, borda count
        retention: Literal["nr", "ar", "dc", "dd"], # Never Retain, Always Retain, Different Class, Degree of Disagreement
        threshold: float = 0.5
    ):
        
        self.sim_metric = sim_metric
        self.k = k
        self.voting = voting
        self.retention = retention
        self.threshold = threshold
        
        self.CD: np.ndarray = None # TODO: think of a more efficient data structure

        self.discrete_cols = []
        self.continuous_cols = []

    # def train_condensation

    # def train_edition
    
    # def train_hybrid

    
    def __train(self, df: pd.DataFrame, red_technique: Literal["None", "MCNN", "AllKNN", "ICF"]="None"):
        data = df.to_numpy()
        if red_technique == 'None':
            self.CD = data
        elif red_technique == 'MCNN': # cridarem les diferents funcions de train segons la reduction technique
            pass
        
        # When loading the train dataset, we can determine whether each column is discrete or continuous
        for i, col in enumerate(df.columns):
            col_element = df[col][0]
            if isinstance(col_element, np.integer):
                self.discrete_cols.append(i)
            elif isinstance(col_element, np.floating):
                self.continuous_cols.append(i)
            else:
                raise ValueError(f"Unkown column type: {type(col_element)}")
            
        
    
    def compute_distance(self, instance: np.ndarray) -> np.ndarray:
        if self.sim_metric == "euc":
            return Metrics.Base.euclidean_dist(self.CD, instance)
        elif self.sim_metric == "cos":
            return Metrics.Base.cosine_dist(self.CD, instance)
        elif self.sim_metric == "heom":
            return self.__heom() # TODO: complete
        elif self.sim_metric == "ivdm":
            return Metrics.IVDM.compute(self.CD, instance, self.discrete_cols, self.continuous_cols) # TODO: complete
        elif self.sim_metric == "gwhsm":
            return self.__gwhsm() # TODO: complete
    
    
    def return_nn(self, ordered_dist):
        return ordered_dist[:self.k, -1]
    
    def modified_plurality(self, nearest_outputs):
        list_app = list(sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True))
        while len(list_app)>1:
            if list_app[0][1] == list_app[1][1]:
                nearest_outputs =  nearest_outputs[:-1]
                list_app = list(sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True))
            else:
                return list_app[0][0]
        if len(list_app)==1:
            return list_app[0][0]

    
    def borda_count(self, nearest_outputs):
        list_bc = [(nearest, len(nearest_outputs)-(i+1)) for i, nearest in enumerate(nearest_outputs)]
        dic_bc = dict.fromkeys(set(nearest_outputs), 0)
        for key, weight in list_bc:
            dic_bc[key] = dic_bc[key] + weight
        list_bc = list(sorted(dic_bc.items(), key=lambda x: x[1], reverse=True))
        tied = [num for i, (num, weight) in enumerate(list_bc) if weight == list_bc[0][1]]
        if len(tied)>1:
            dict_app = dict(list(sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True)))
            max_ = 0
            tied2bool = False
            tied2 = []
            for num in tied:
                if dict_app[num]>max_:
                    max_ = dict_app[num]
                    res = num
                    tied2bool = False
                elif dict_app[num] == max_:
                    tied2bool = True
                    tied2.append(num)
            if tied2bool == False:
                return res
            else:
                min_ = nearest_outputs.index(tied2[0])
                selected = tied2[0]
                for i in range(1,len(tied2)):
                    pos = nearest_outputs.index(tied2[i])
                    if pos<min_:
                        min_ = pos
                        selected = tied2[i]
                return selected
        else:
            return list_bc[0][0]
        
    
    def voting_schema(self, nearest_outputs):
        if self.voting == 'mp': # modified plurality
            return self.modified_plurality(nearest_outputs)
        elif self.voting == 'bc': # borda count
            return self.borda_count(nearest_outputs)
    
    
    def update_cd(self, instance, output, nearest_outputs):
        if self.retention == "nr":
            return
        
        elif self.retention == "ar":
            
            self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        elif self.retention == "dc":
            if output != instance[-1]:
                self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        elif self.retention == "dd":
            dict_app = dict(list(sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True)))
            tied = [list(dict_app.keys())[0]]
            max_ = dict_app[list(dict_app.keys())[0]]
            for key in list(dict_app.keys())[1:]:
                if dict_app[key]==max_:
                    tied.append(key)
                else:
                    break
            if len(tied)>1:
                min_ = nearest_outputs.index(tied[0])
                majority_class = tied[0]
                for i in range(1,len(tied)):
                    pos = nearest_outputs.index(tied[i])
                    if pos<min_:
                        min_ = pos
                        majority_class = tied[i]
            else:
                majority_class = tied[0]
            num_majority_class = dict_app[majority_class]
            num_remaining_cases = len(nearest_outputs)-num_majority_class
            num_classes = len(dict_app)
            if (num_classes-1)==0:
                d = 0
            else:
                d = num_remaining_cases / ((num_classes-1) * num_majority_class)
            if d>= self.threshold:
                self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)

    def wilson_editing_v1(self, X, y):
        knn = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean').fit(X, y)  # self.k
        y_pred = knn.predict(X)
        mask = np.where(y_pred != y)
        X_r = np.delete(X, mask, axis=0)
        y_r = np.delete(y, mask, axis=0)
        return X_r, y_r

    def wilson_editing_v2(self, X, y):
        y_pred = []
        for _, instance in X.iterrows():
            instance = instance.to_numpy()
            nearest_outputs = self.return_nn(Metrics.Base.euclidean_dist(X, instance))
            output = self.modified_plurality(nearest_outputs)
            y_pred.append(output)
        mask = np.where(y_pred != y)
        X_r = np.delete(X, mask, axis=0)
        y_r = np.delete(y, mask, axis=0)
        return X_r, y_r

    def reachable(self, X, y):
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

        return np.array(recheable_list)

    def coverage(self, reachable):
        coverage_array = [[] for i in range(len(reachable))]
        for i, reachable_ in enumerate(reachable):
            for reach in reachable_:
                if i not in coverage_array[reach]:
                    coverage_array[reach].append(i)
        return np.array(coverage_array)

    def ICF(self, X, y):
        X, y = self.wilson_editing_v2(X, y)
        reachable = self.reachable(X, y)
        coverage = self.coverage(reachable)
        progress = False
        while not progress:
            X_r = X.copy()
            y_r = y.copy()
            reachable_r = reachable.copy()
            coverage_r = coverage.copy()
            for i, x in enumerate(X):
                reachable = 0
                coverage = 0
                if len(reachable[i]) > len(coverage[i]):
                    X_r = np.delete(X_r, x, axis=0)
                    y_r = np.delete(y_r, x, axis=0)
                    reachable_r = np.delete(reachable_r, x, axis=0)
                    coverage_r = np.delete(coverage_r, x, axis=0)
                else:
                    progress = True
            X = X_r.copy()
            y = y_r.copy()
            reachable = reachable_r.copy()
            coverage = coverage_r.copy()
        return X_r, y_r

    def pca_analysis(self, X, y, n_components):
        pca = PCA(n_components=n_components)
        X_r = pca.fit_transform(X)
        principal_Df = pd.DataFrame(data=X_r
                                    , columns=['principal component 1', 'principal component 2'])
        if n_components == 2:
            plt.figure()
            plt.figure(figsize=(15, 15))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=14)
            plt.xlabel('Principal Component - 1', fontsize=20)
            plt.ylabel('Principal Component - 2', fontsize=20)
            plt.title("Principal Component Analysis", fontsize=20)
            targets = set(y)
            for target in targets:
                indicesToKeep = y == target
                plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1']
                            , principal_Df.loc[indicesToKeep, 'principal component 2'], s=50)

            plt.legend(targets, prop={'size': 15}, loc='upper right')
        return X_r
        
    """ def ir_KIBLAlgorithm(...) # kibl amb les diferents instance reduction techniques
        # una reducció
        self.__train(train_df)
        self.__kIBLAlgorithm()
    
    def KIBLAlgorithm(...) # "substitueix" a la kiblalgorithm que teniem abans
        # cap reducció
        self.__train(train_df)
        self.__kIBLAlgorithm() """
    
    
    
    # def __kIBLAlgorithm(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[int]:
    def kIBLAlgorithm(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[int]:
        self.__train(train_df)
        
        predictions = []
        for _, instance in test_df.iterrows():
            instance = instance.to_numpy()
            # Compute similarity metric -> important no passar ultima columna (o la de la classe).
            # Obtain k-nearest neighbors
            nearest_outputs = self.return_nn(self.compute_distance(instance))
            # Decide output based on voting scheme
            output = self.voting_schema(nearest_outputs)
            
            predictions.append(output)
            
            # Update CD based on retention policy
            self.update_cd(instance, output, nearest_outputs)
        
        return predictions
        
    def predict(self, X: pd.DataFrame):
        # Perhaps not used
        pass



if __name__ == "__main__":
        
    folds = 10  
    

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
    
    # a = [['ivdm'],["mp"],[3], ['nr']] #FIXME: add missing cases (now they are not completely implemented)
    a = [['euc','cos'],["mp"],[3,5,7], ['nr', 'ar', 'dc']] #FIXME: add missing cases (now they are not completely implemented)
    parameters_combinations = list(itertools.product(*a))

    for dataset in ["credit-a", "pen-based"]:
        results = {}
        for metric, voting, k, retention in parameters_combinations:
            test_name = f"{metric}_{voting}_{k}_{retention}"
            ibl_learner = KIBLearner(
                sim_metric=metric,
                k=k,
                voting=voting,
                retention=retention
            )
            total_accuracy = 0
            results[test_name] = {}
            for i in range(folds):
                df_train = pd.read_csv(f"preprocessed/{dataset}/{dataset}.fold.{str(i).zfill(6)}.train.csv")
                df_test = pd.read_csv(f"preprocessed/{dataset}/{dataset}.fold.{str(i).zfill(6)}.test.csv")
                y_pred = ibl_learner.kIBLAlgorithm(df_train, df_test)
                y_true = df_test[df_test.columns[-1]]

                results[test_name][i] = {}

                results[test_name][i]["y_true"] = y_true.to_list()
                results[test_name][i]["y_pred"] = y_pred
                
                correct = 0
                for pred, true in zip(y_pred, y_true):
                    if pred == true: correct += 1
                    # print(f"Prediction: {pred}. True value: {true}")
                curr_accuracy = (correct / len(y_pred))
                results[test_name][i]["fold_accuracy"] = curr_accuracy
                total_accuracy += correct
                print(f"Accuracy fold {i}: {round(curr_accuracy * 100, 4)}%")
            
            total_accuracy /= (len(y_pred)*folds)
            results[test_name]["total_accuracy"] = total_accuracy
            print(f"Total accuracy: {round(total_accuracy * 100, 4)}%")
        
        
        with open(f"results_{dataset}.json", "w+") as f:
            json.dump(results, f)
