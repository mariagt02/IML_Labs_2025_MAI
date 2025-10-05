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
        
        self.CD: np.ndarray = None # TODO: think of a more efficient data structure

        self.discrete_cols = []
        self.continuous_cols = []
    
    def __euclidean_dist(self, instance: np.ndarray):
        """
        Returns (ordered_dist): the list of all instances ordered from smallest to largest distance to the given instance.
        """
        
        instance_repeated = np.tile(instance, (self.CD.shape[0], 1))
        euclidean_dist = np.linalg.norm(instance_repeated[:, :-1] - self.CD[:, :-1], axis=1)
        
        sorted_idx = np.argsort(euclidean_dist)
        
        return self.CD[sorted_idx]
    
    def __cosine_dist(self, instance: np.ndarray):
        """
        Returns (ordered_dist): the list of all instances ordered from smallest to largest distance to the given instance.
        """
        instance_repeated = np.tile(instance, (self.CD.shape[0], 1))
        cosine_dist = 1 - (instance_repeated[:, :-1] @ self.CD[:, :-1].T) / (np.linalg.norm(instance_repeated[:, :-1], axis=1) * np.linalg.norm(self.CD[:, :-1], axis=1))
        
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

        
    
    def __heom(self, x1, x2):
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia 
        pass
    
    def __gwhsm(self, x1, x2): #FIXME: input ??
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
        if not self.CD:
            self.__train(train_df)
        
        for _, instance in test_df.iterrows():
            self.__cosine_dist(instance.to_numpy())
            # Compute similarity metric -> important no passar ultima columna (o la de la classe).
            # Obtain k-nearest neighbors
            # Decide output based on voting scheme
            # Update CD based on retention policy
            break
        
        
    def predict(self, X: pd.DataFrame):
        # Perhaps not used
        pass



if __name__ == "__main__":
    df_train = pd.read_csv("preprocessed/credit-a/credit-a.fold.000000.train.csv")
    df_test = pd.read_csv("preprocessed/credit-a/credit-a.fold.000000.test.csv")
    ibl_learner = KIBLearner(
        sim_metric="euc",
        k=3,
        voting="mp",
        retention="nr"
    )
    ibl_learner.kIBLAlgorithm(df_train, df_test)
    
    