import pandas as pd
import numpy as np
from typing import Literal
from collections import Counter
from metrics import Metrics
from enum import Enum

class IBLHyperParameters:
    class SimMetrics(str, Enum):
        EUCLIDEAN = "euc"
        COSINE = "cos"
        IVDM = "ivdm"
        HEOM = "heom"
        GWHSM = "gwhsm"
    
    class Voting(str, Enum):
        # The voting scheme
        MODIFIED_PLURALITY = "mp"
        BORDA_COUNT = "bc"
    
    class Retention(str, Enum):
        # The retention policy
        NEVER_RETAIN = "nr"
        ALWAYS_RETAIN = "ar"
        DIFFERENT_CLASS = "dc"
        DEGREE_OF_DISAGREEMENT = "dd"
    
    @classmethod
    def get_all_values(cls, exclude = [SimMetrics.IVDM, SimMetrics.HEOM, SimMetrics.GWHSM]) -> list[list[str]]:
        # We exclude the distance metrics that are not implemented
        results = []
        for _, obj in cls.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, Enum):
                results.append([member.value for member in obj if member not in exclude])
        
        return results
    

class KIBLearner:
    def __init__(self,
        sim_metric: IBLHyperParameters.SimMetrics,
        k: int,
        voting: IBLHyperParameters.Voting,
        retention: IBLHyperParameters.Retention,
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

    # def train_condensation -> MCNN
    def _get_class_data(self, data: np.ndarray):
        class_data = {}
        if data.size == 0:
            return {}, []
        
        labels = np.unique(data[:, -1])
        for label in labels:
            class_data[label] = data[data[:, -1] == label]
        return class_data, labels 
    
    def _get_centroid(self, data: np.ndarray):
        if data.size == 0:
            return None
        return np.mean(data[:, :-1], axis=0)
    
    def _nearest_neighbor_label(self, sample: np.ndarray, prototypes: np.ndarray):
        if prototypes.size == 0 or prototypes.ndim != 2:
            return None 
        sample_features = sample[:-1]
        prototype_features = prototypes[:, :-1]
        distances = np.sum(np.square(prototype_features - sample_features), axis=1)
        nearest_index = np.argmin(distances)
        return prototypes[nearest_index, -1]

    def _find_closest_sample_to_centroid(self, data: np.ndarray, centroid: np.ndarray):
        if centroid is None or data.size == 0:
            return None
        features = data[:, :-1]
        distances = np.linalg.norm(features - centroid, axis=1)
        closest_index = np.argmin(distances)
        return data[closest_index]
    
    def _train_condensation(self, data: np.ndarray) -> np.ndarray:
    # NOTA: Utilitzo mateixa notació que en el pseudocodi del paper
        T = data     
        Q = np.empty((0, T.shape[1]))
        S_current = T
        while True:
            t = 0
            S_new = S_current
            P_current = np.empty((0, T.shape[1]))
            # Repeats until S_new (the misclassified set S'') is empty (Step 11).
            while True:
                t += 1 
                class_data_s_new, labels = self._get_class_data(S_new) 
                new_prototypes = []
                for label in labels:
                    data_j = class_data_s_new[label]
                    
                    if data_j.size > 0:
                        centroid_j = self._get_centroid(data_j) 
                        prototype_j = self._find_closest_sample_to_centroid(data_j, centroid_j)
                        
                        if prototype_j is not None:
                            new_prototypes.append(prototype_j)
                            
                P = np.array(new_prototypes) 
                
                if P.size == 0:
                    break 
                P_current = np.vstack([P_current, P]) 
                
                S_correctly_classified = np.empty((0, T.shape[1])) # S' (Step 8)
                S_misclassified = np.empty((0, T.shape[1]))      # S'' (Step 8)
                for sample in S_new:
                    if Q.size == 0 and P_current.size == 0:
                        break   
                    nn_label = self._nearest_neighbor_label(sample, P_current) 
                    if nn_label != sample[-1]:
                        S_misclassified = np.vstack([S_misclassified, sample])
                    else:
                        S_correctly_classified = np.vstack([S_correctly_classified, sample])
                        
                S_new = S_misclassified 
                if S_misclassified.size == 0:
                    break # all samples in the current S_new are correctly classified
            Q = np.vstack([Q, P_current])
            
            S_misclassified_final = np.empty((0, T.shape[1]))
            
            # check consistency of the ENTIRE training set T using Q
            for sample in T:
                nn_label = self._nearest_neighbor_label(sample, Q)
                if nn_label != sample[-1]:
                    S_misclassified_final = np.vstack([S_misclassified_final, sample])
            S_current = S_misclassified_final
            
            if S_misclassified_final.size == 0:
                return Q

    # def train_edition
    
    # def train_hybrid

    
    def __train(self, df: pd.DataFrame, red_technique: Literal["None", "MCNN", "AllKNN", "ICF"]="None"):
        data = df.to_numpy()
        if red_technique == 'None':
            self.CD = data
        elif red_technique == 'MCNN': # cridarem les diferents funcions de train segons la reduction technique
            self.CD = self._train_condensation(data)
        
        # When loading the train dataset, we can determine whether each column is discrete or continuous
        for i, col in enumerate(df.columns[:-1]):
            col_element = df[col][0]
            if isinstance(col_element, np.integer):
                self.discrete_cols.append(i)
            elif isinstance(col_element, np.floating):
                self.continuous_cols.append(i)
            else:
                raise ValueError(f"Unkown column type: {type(col_element)}")
            
        if self.sim_metric == IBLHyperParameters.SimMetrics.IVDM:
            self.ivdm_metric = Metrics.IVDM(self.CD, self.discrete_cols, self.continuous_cols)
    
    def compute_distance(self, instance: np.ndarray) -> np.ndarray:
        if self.sim_metric == IBLHyperParameters.SimMetrics.EUCLIDEAN:
            return Metrics.Base.euclidean_dist(self.CD, instance)
        elif self.sim_metric == IBLHyperParameters.SimMetrics.COSINE:
            return Metrics.Base.cosine_dist(self.CD, instance)
        elif self.sim_metric == IBLHyperParameters.SimMetrics.HEOM:
            return self.__heom() # TODO: complete
        elif self.sim_metric == IBLHyperParameters.SimMetrics.IVDM:
            return self.ivdm_metric.compute(instance)
        elif self.sim_metric == IBLHyperParameters.SimMetrics.GWHSM:
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
        if self.voting == IBLHyperParameters.Voting.MODIFIED_PLURALITY:
            return self.modified_plurality(nearest_outputs)
        elif self.voting == IBLHyperParameters.Voting.BORDA_COUNT:
            return self.borda_count(nearest_outputs)
    
    
    def update_cd(self, instance, output, nearest_outputs):
        if self.retention == IBLHyperParameters.Retention.NEVER_RETAIN:
            return
        
        elif self.retention == IBLHyperParameters.Retention.ALWAYS_RETAIN:
            
            self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        elif self.retention == IBLHyperParameters.Retention.DIFFERENT_CLASS:
            if output != instance[-1]:
                self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        elif self.retention == IBLHyperParameters.Retention.DEGREE_OF_DISAGREEMENT:
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
            d = num_remaining_cases / ((num_classes-1) * num_majority_class) # FIXME: division by zero sometimes
            if d>= self.threshold:
                self.CD = np.append(self.CD, instance.reshape(1, -1), axis=0)
        
        
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

