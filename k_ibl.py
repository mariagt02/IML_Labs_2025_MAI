import pandas as pd
import numpy as np
from typing import Literal
import math
from collections import Counter
import json
import itertools
from sklearn.neighbors import KNeighborsClassifier

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
    
    def __euclidean_dist(self, instance: np.ndarray) -> np.ndarray:
        """
        Returns (ordered_dist): the list of all instances ordered from smallest to largest distance to the given instance.
        """
        
        instance_repeated = np.tile(instance, (self.CD.shape[0], 1))
        euclidean_dist = np.linalg.norm(instance_repeated[:, :-1] - self.CD[:, :-1], axis=1)
        
        sorted_idx = np.argsort(euclidean_dist)
        
        return self.CD[sorted_idx]
    
    def __cosine_dist(self, instance: np.ndarray) -> np.ndarray:
        """
        Returns (ordered_dist): the list of all instances ordered from smallest to largest distance to the given instance.
        """
        instance_repeated = np.tile(instance, (self.CD.shape[0], 1))
        cosine_dist = 1 - (np.einsum('ij,ij->i', instance_repeated[:, :-1], self.CD[:, :-1]) / (np.linalg.norm(instance_repeated[:, :-1], axis=1) * np.linalg.norm(self.CD[:, :-1], axis=1)))

        sorted_idx = np.argsort(cosine_dist)
        

        return self.CD[sorted_idx]
        
    
    
    def __vdm(self, x: int, y: int, a: int): # Eq (8)
        res = 0
        q = 2 # It could also take value 1 TODO: decide which of the two values to use
        
        for c in len(np.unique(self.CD[:, -1])): 
            # num. instances in training set that have value x for attribute a
            N_ax = np.sum(self.CD[:, a] == x)
            # num. of instances in the training set that have value x for attribute a and output class c
            N_axc = np.sum((self.CD[:, a] == x) & (self.CD[:, -1] == c))
            
            N_ay = np.sum(self.CD[:, a] == y)
            N_ayc = np.sum((self.CD[:, a] == y) & (self.CD[:, -1] == c))

            P_axc = N_axc / N_ax # conditional probability that the output class is "c" given that attribute "a" has value "x"
            P_ayc = N_ayc / N_ay # conditional probability that the output class is "c" given that attribute "a" has value "y"
            
            res += np.abs(P_axc - P_ayc) ** q
            
        return res
    
    def __discretize(self, x: int | float, a: int): # Eq (18)
        s = max(5, len(np.unique(self.CD[:, -1]))) 
        if a in self.discrete_cols:
            return x
        elif x == np.max(self.CD[:, a]):
            return s
        else:
            w_a = np.abs(max(self.CD[:, a]) - min(self.CD[:, a])) / s # Eq (17)
            return np.floor((x - np.min(self.CD[:, a])) / w_a) + 1
            
    
    def __mid_point(self, u: int, a: int): # Eq (24)
        s = max(5, len(np.unique(self.CD[:, -1])))
        width_a = (max(self.CD[:, a]) - min(self.CD[:, a])) / s  # eq 17

        return min(self.CD[:, a]) + (width_a * u) + (width_a * 0.5)

    
    def __learn_P(self, a: int, x: float, c: int): # figure 5
        # TODO: complete
        pass
    
    
    def __interpolated_prob(self, x: float, a: int, c: int): # Eq (23)
        res = 0
        
        u = self.__discretize(x, a) # Determining the discrete value for attribute "a" correspodning to value "x".
        
        mid_au = self.__mid_point(u, a)
        if x < mid_au:
            # "The value of u is found by first setting u = discretize_a(x), and then substracting 1 from u if x < mid_au" 
            u -= 1
            mid_au = self.__mid_point(u, a)
        
        mid_au1 = self.__mid_point(u + 1, a)
        
        P_auc = 0
        P_au1c = 0
        
        
        return res
    
    def __ivdm_a(self, x: int | float, y: int | float, a: int): # x and y are values of the attributes. Eq (22)
    
        if a in self.continuous_cols:
            # Eq (22.b)
            res = 0
            for c in len(np.unique(self.C[:, -1])):
                p_acx = self.__interpolated_prob(x, a, c)
                p_acy = self.__interpolated_prob(y, a, c)
                res += np.abs(p_acx - p_acy) ** 2
        elif a in self.discrete_cols:
            # VDM --> paper page 6
            res = self.__vdm(x, y, a) # Eq (22.a)
        
        return res
    
    
    def __ivdm(self, x: np.ndarray, y: np.ndarray): # Eq (21)
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia
        for a in range(x.shape[1]):
            self.__ivdm_a(x[a], y[a], a)

        
    
    def __heom(self, x1, x2) -> np.ndarray:
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia 
        pass
    
    def __gwhsm(self, x1, x2) -> np.ndarray: #FIXME: input ??
       # TODO: perhaps remove?
        pass
    
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
            return self.__euclidean_dist(instance)
        elif self.sim_metric == "cos":
            return self.__cosine_dist(instance)
        elif self.sim_metric == "heom":
            return self.__heom() # TODO: complete
        elif self.sim_metric == "ivdm":
            return self.__ivdm() # TODO: complete
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
            d = num_remaining_cases / ((num_classes-1) * num_majority_class)
            if d>= self.threshold:
                self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
    
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