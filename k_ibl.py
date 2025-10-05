import pandas as pd
import numpy as np
from typing import Literal
import math
from collections import Counter

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
        
        self.CD = [] # TODO: think of a more efficient data structure
    
    def __euclidean_dist(self, x1, x2):
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia 
        # treballar en matrius, no loop.
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2))) 
    
    def __cosine_dist(self, x1, x2):
        # idem que euclidean_dist
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    
    def __ivdm(self, x1, x2): # FIXME: input ??
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia 
        pass
    
    def __heom(self, x1, x2):
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia 
        pass
    
    def __gwhsm(self, x1, x2): #FIXME: input ??
       # TODO: perhaps remove?
        pass
    
    def __train(self, df: pd.DataFrame):
        # Load training dataset into CD (considering IB1)
        pass
    
    
    def compute_distance(self, instance):
        if self.sim_metric == "euc":
            return self.__euclidean_dist()
        elif self.sim_metric == "cos":
            return self.__cosine_dist()
        elif self.sim_metric == "heom":
            return self.__heom()
        elif self.sim_metric == "ivdm":
            return self.__ivdm()   
        elif self.sim_metric == "gwhsm":
            return self.__gwhsm()
    
    
    def return_nn(self, ordered_dist):
        return ordered_dist[:self.k, -1]
    
    def modified_plurality(self, nearest_outputs):
        mp = False
        list_app = list(sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True))
        while not mp and len(list_app)>1:
            if list_app[0][1] == list_app[1][1]:
                nearest_outputs =  nearest_outputs[:-1]
                list_app = list(sorted(Counter(nearest_outputs).items(), key=lambda x: x[1], reverse=True))
            else:
                mp = True
                return list_app[0][0]
        if len(list_app)==1:
            return list_app[0][0]

    
    def borda_count(self, nearest_outputs):
        list_bc = [(nearest, len(nearest_outputs)-(i+1)) for i, nearest in enumerate(nearest_outputs)]
        dic_bc = dict.fromkeys(set(nearest_outputs), 0)
        for key, weight in list_bc:
            dic_bc[key] = dic_bc[key] + weight
        list_bc = list(sorted(dic_bc.items(), key=lambda x: x[1], reverse=True))
        if len(list_bc)> 1:
            if list_bc[0][1] == list_bc[1][1]:
                pass            
            else:
                return list_bc[0][0]
        else:
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
            pass
        elif self.retention == "dc":
            pass
        elif self.retention == "dd":
            pass
    
    
    def kIBLAlgorithm(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        if len(self.CD) == 0:
            self.__train(train_df)
        
        for _, instance in test_df.iterrows():
            # Compute similarity metric -> important no passar ultima columna (o la de la classe).
            # Obtain k-nearest neighbors
            nearest_outputs = self.return_nn(self.compute_distance(instance))
            # Decide output based on voting scheme
            # Update CD based on retention policy
            pass
        
        
    def predict(self, X: pd.DataFrame):
        # Perhaps not used
        pass



if __name__ == "__main__":
    learner = KIBLearner()
    
    