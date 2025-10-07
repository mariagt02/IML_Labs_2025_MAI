import numpy as np

__all__ = ["IVDM", "HEOM", "GWHSM"]

class IVDM:
    
    @staticmethod
    def compute(
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        discrete_cols: list[int],
        continuous_cols: list[int]
    ) -> float: # Eq (21)
        
        total = 0
        # RETURN (ordered_dist): llista amb totes les instancies de menor a major distancia
        for a in range(x.shape[1]):
            total += IVDM.__ivdm_a(x[a], y[a], a, data, discrete_cols, continuous_cols)
        
        return total
    
    
    @staticmethod
    def __ivdm_a(
        x: int | float,
        y: int | float,
        a: int,
        data: np.ndarray,
        discrete_cols: list[int],
        continuous_cols: list[int]
    ): # x and y are values of the attributes. Eq (22)
    
        if a in continuous_cols:
            # Eq (22.b)
            res = 0
            for c in len(np.unique(data[:, -1])):
                p_acx = IVDM.__interpolated_prob(x, a, c, data, discrete_cols)
                p_acy = IVDM.__interpolated_prob(y, a, c)
                res += np.abs(p_acx - p_acy) ** 2
        elif a in discrete_cols:
            # VDM --> paper page 6
            res = IVDM.__vdm(x, y, a, data) # Eq (22.a)
        
        return res
    
    
    @staticmethod
    def __vdm(x: int, y: int, a: int, data: np.ndarray): # Eq (8)
        res = 0
        q = 2 # It could also take value 1 TODO: decide which of the two values to use
        
        for c in len(np.unique(data[:, -1])): 
            # num. instances in training set that have value x for attribute a
            N_ax = np.sum(data[:, a] == x)
            # num. of instances in the training set that have value x for attribute a and output class c
            N_axc = np.sum((data[:, a] == x) & (data[:, -1] == c))
            
            N_ay = np.sum(data[:, a] == y)
            N_ayc = np.sum((data[:, a] == y) & (data[:, -1] == c))

            P_axc = N_axc / N_ax # conditional probability that the output class is "c" given that attribute "a" has value "x"
            P_ayc = N_ayc / N_ay # conditional probability that the output class is "c" given that attribute "a" has value "y"
            
            res += np.abs(P_axc - P_ayc) ** q
            
        return res
    
    @staticmethod
    def __interpolated_prob(x: float, a: int, c: int, data: np.ndarray, discrete_cols: list[int]): # Eq (23)
        res = 0
        
        u = IVDM.__discretize(x, a, data, discrete_cols) # Determining the discrete value for attribute "a" correspodning to value "x".
        
        mid_au = IVDM.__mid_point(u, a, data)
        if x < mid_au:
            # "The value of u is found by first setting u = discretize_a(x), and then substracting 1 from u if x < mid_au" 
            u -= 1
            mid_au = IVDM.__mid_point(u, a, data)
        
        mid_au1 = IVDM.__mid_point(u + 1, a, data)
        
        P_auc = 0
        P_au1c = 0
        
        return res
    
    
    @staticmethod
    def __discretize(x: int | float, a: int, data: np.ndarray, discrete_cols: list[int]): # Eq (18)
        s = max(5, len(np.unique(data[:, -1]))) 
        if a in discrete_cols:
            return x
        elif x == np.max(data[:, a]):
            return s
        else:
            w_a = np.abs(max(data[:, a]) - min(data[:, a])) / s # Eq (17)
            return np.floor((x - np.min(data[:, a])) / w_a) + 1
            
    
    @staticmethod
    def __mid_point(u: int, a: int, data: np.ndarray): # Eq (24)
        s = max(5, len(np.unique(data[:, -1])))
        width_a = (max(data[:, a]) - min(data[:, a])) / s  # eq 17

        return min(data[:, a]) + (width_a * u) + (width_a * 0.5)

    
    @staticmethod
    def __learn_P(a: int, x: float, c: int): # figure 5
        # TODO: complete
        pass
    
    
    
    
    
    


class HEOM:
    # TODO: complete if the metric wants to be implemented
    pass

class GWHSM:
    # TODO: complete if the metric wants to be implemented
    pass