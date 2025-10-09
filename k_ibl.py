import pandas as pd
import numpy as np
from typing import Literal
import math
from collections import Counter
import json
import itertools
from sklearn.neighbors import KNeighborsClassifier
from metrics import Metrics

class KIBLearner:
    def __init__(self,
        sim_metric: Literal["euc", "cos", "heom", "ivdm", "gwhsm"],
        k: int,
        voting: Literal["mp", "bc"], # modified plurality, borda count
        retention: Literal["nr", "ar", "dc", "dd"] # Never Retain, Always Retain, Different Class, Degree of Disagreement
    ):
        
        self.sim_metric = sim_metric
        self.k = k
        self.voting = voting
        self.retention = retention
        
        self.CD: np.ndarray = None # TODO: think of a more efficient data structure

        self.discrete_cols = []
        self.continuous_cols = []
    
    
    def __train(self, df: pd.DataFrame):
        self.CD = df.to_numpy()
        
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
        # TODO tenir en compte casos en que hi ha empats.
        # if len(list_bc)> 1:
        #     if list_bc[0][1] == list_bc[1][1]:
        #         dic_app = (sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True))
        #         if dic_app[list_bc[0][0]] == dic_app[list_bc[1][0]]:
                    
        #         else:
                             
        #     else:
        #         return list_bc[0][0]
        # else:
        return list_bc[0][0]

    
    def voting_schema(self, nearest_outputs):
        if self.voting == 'mp': # modified plurality
            return self.modified_plurality(nearest_outputs)
        elif self.voting == 'bc': # borda count
            return self.borda_count(nearest_outputs)
    
    
    def update_cd(self, instance, output):
        if self.retention == "nr":
            return
        
        elif self.retention == "ar":
            
            self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        elif self.retention == "dc":
            if output != instance[-1]:
                self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        elif self.retention == "dd":
            pass
    
    
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
            self.update_cd(instance, output)
        
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
    
    a = [['ivdm'],["mp"],[3], ['nr']] #FIXME: add missing cases (now they are not completely implemented)
    # a = [['euc','cos'],["mp"],[3,5,7], ['nr', 'ar', 'dc']] #FIXME: add missing cases (now they are not completely implemented)
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