import numpy as np
from collections import defaultdict

__all__ = ["IVDM", "HEOM", "GWHSM"]

class IVDM:
    
    def __init__(self, data: np.ndarray, discrete_cols: list[int], continuous_cols: list[int]):
        
        self.data = data
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols
        
        # Compute p matrix only once when the class is initalized
        self.P_avc = self.__learn_P()
    
    def compute(
        self,
        instance: np.ndarray,
    ) -> np.ndarray: # Eq (21) // retornem llista amb els punts del CD ordenats de menor a major distància respecte a 'instance'
        
        ivdm_distances = np.zeros(len(self.data)) # llista amb les distàncies de cada punt del CD a la instance concreta
        
        for idx, point in enumerate(self.data):
            # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia
            for a in range(instance.shape[0]):
                ivdm_distances[idx] += self.__ivdm_a(point[a], instance[a], a)

        sorted_idx = np.argsort(ivdm_distances) # ordenem les instàncies segons les distàncies obtingudes
        
        return self.data[sorted_idx] 
            
    
    def __ivdm_a(
        self,
        x: int | float,
        y: int | float,
        a: int
    ): # x and y are values of the attributes. Eq (22)
    
        if a in self.continuous_cols:
            # Eq (22.b)
            res = 0
            for c in range(len(np.unique(self.data[:, -1]))):
                p_acx = self.__interpolated_prob(x, a, c)
                p_acy = self.__interpolated_prob(y, a, c)
                res += np.abs(p_acx - p_acy) ** 2
        elif a in self.discrete_cols:
            # VDM --> paper page 6
            res = self.__vdm(x, y, a) # Eq (22.a)
        
        return res
    
    
    
    def __vdm(self, x: int, y: int, a: int): # Eq (8)
        res = 0
        q = 2 # It could also take value 1 TODO: decide which of the two values to use
        
        for c in range(len(np.unique(self.data[:, -1]))): 
            # num. instances in training set that have value x for attribute a
            N_ax = np.sum(self.data[:, a] == x)
            # num. of instances in the training set that have value x for attribute a and output class c
            N_axc = np.sum((self.data[:, a] == x) & (self.data[:, -1] == c))
            
            N_ay = np.sum(self.data[:, a] == y)
            N_ayc = np.sum((self.data[:, a] == y) & (self.data[:, -1] == c))

            P_axc = N_axc / N_ax # conditional probability that the output class is "c" given that attribute "a" has value "x"
            P_ayc = N_ayc / N_ay # conditional probability that the output class is "c" given that attribute "a" has value "y"
            
            res += np.abs(P_axc - P_ayc) ** q
            
        return res
    
    
    def __interpolated_prob(self, x: float, a: int, c: int): # Eq (23)
        res = 0
        
        u = self.__discretize(x, a) # Determining the discrete value for attribute "a" correspodning to value "x".
        
        mid_au = self.__mid_point(u, a)
        if x < mid_au:
            # "The value of u is found by first setting u = discretize_a(x), and then substracting 1 from u if x < mid_au" 
            u -= 1
            mid_au = self.__mid_point(u, a)
        
        mid_au1 = self.__mid_point(u + 1, a)
        
        P_auc = self.P_avc[f"{a}_{u}_{c}"]
        P_au1c = self.P_avc[f"{a}_{u + 1}_{c}"] # Pray that this entry exists
        
        res = P_auc + ((x - mid_au) / (mid_au1 - mid_au)) * (P_au1c - P_auc)
        
        return res
    
    
    
    def __discretize(self, x: int | float, a: int): # Eq (18)
        s = max(5, len(np.unique(self.data[:, -1]))) 
        if a in self.discrete_cols:
            return x
        elif x == np.max(self.data[:, a]):
            return s
        else:
            w_a = np.abs(max(self.data[:, a]) - min(self.data[:, a])) / s # Eq (17)
            return np.floor((x - np.min(self.data[:, a])) / w_a) + 1
            
    
    def __mid_point(self, u: int, a: int): # Eq (24)
        s = max(5, len(np.unique(self.data[:, -1])))
        width_a = (max(self.data[:, a]) - min(self.data[:, a])) / s  # eq 17

        return min(self.data[:, a]) + (width_a * u) + (width_a * 0.5)

    
    
    def __learn_P(self): # figure 5
        n_avc = defaultdict(int)
        n_av = defaultdict(int)
        p_avc = defaultdict(int)
        
        for a in range(self.data.shape[1]):
            v_values_attribute = []
            for instance in self.data:
                x = instance[a]
                v = self.__discretize(x, a) # in which "bin" the value x of attribute a falls into
                v_values_attribute.append(v) # for each attribute, we save which "bins" we have obtained
                c = instance[-1]
                n_avc[f"{a}_{v}_{c}"] += 1 # We implement it as a dictionary because we do not know how to know in advance the possible values of v
                n_av[f"{a}_{v}"] += 1
            
            
            for v in np.unique(v_values_attribute):
                for c in np.unique(self.data[:, -1]):
                    if n_av[f"{a}_{v}"] == 0:
                        p_avc[f"{a}_{v}_{c}"] = 0
                    else:
                        p_avc[f"{a}_{v}_{c}"] = n_avc[f"{a}_{v}_{c}"] / n_av[f"{a}_{v}"]
        
        return p_avc
            
    

class HEOM:
    # TODO: complete if the metric wants to be implemented
    pass

class GWHSM:
    # TODO: complete if the metric wants to be implemented
    pass