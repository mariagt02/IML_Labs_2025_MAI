import pandas as pd
import numpy as np
from typing import Literal
import math

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
    
    def voting_schema(self, nearest_outputs):
        if self.voting == 'mp': # modified plurality
            pass
        elif self.voting == 'bc': # borda count
            pass
        return 0
    
    
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
            # Decide output based on voting scheme
            # Update CD based on retention policy
            pass
        
        
    def predict(self, X: pd.DataFrame):
        # Perhaps not used
        pass



if __name__ == "__main__":
    learner = KIBLearner()
    
    