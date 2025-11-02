import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
import time

__all__ = ["IVDM", "HEOM", "GWHSM"]

class IVDM:
    
    def __init__(self, data: np.ndarray, discrete_cols: list[int], continuous_cols: list[int]):
        
        self.data = data
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols
        
        # Compute p matrix only once when the class is initalized
        self.P_avc = self.__learn_P()

        # build cached statistics and vectorized helpers to speed up distance computation
        self.__build_cache()
    
    def compute(
        self,
        instance: np.ndarray,
    ) -> np.ndarray: # Eq (21) // retornem llista amb els punts del CD ordenats de menor a major distància respecte a 'instance'
        
        # Vectorized computation of IVDM distances between `instance` and all dataset rows.
        
        num_instances = self.data.shape[0]
        ivdm_distances = np.zeros(num_instances, dtype=float)

        # For continuous attributes: use cached reults and per-point class prob arrays and compute
        # contribution = sum_c |p_ac(x_i)-p_ac(y)|^2
        for a in self.continuous_cols:
    
            p_ac_all = self.continuous_Pac[a]
            p_acy = self._interpolated_prob_array(np.array([instance[a]], dtype=float), a)[0]

            contribution = np.sum((p_ac_all - p_acy) ** 2, axis=1)
            ivdm_distances += contribution

        # For discrete attributes: use cached per-point class-prob arrays and compute VDM
        q = 2
        for a in self.discrete_cols:
            p_ac_all = self.discrete_Pvc[a]  # shape (num_instances, n_classes)
            p_acy = self.discrete_value_Pvc[a].get(instance[a], np.zeros(self.num_classes))
            contribution = np.sum(np.abs(p_ac_all - p_acy) ** q, axis=1)
            ivdm_distances += contribution

        sorted_idx = np.argsort(ivdm_distances)
        return self.data[sorted_idx]
            
    
    # def __ivdm_a(
    #     self,
    #     x: int | float,
    #     y: int | float,
    #     a: int
    # ): # x and y are values of the attributes. Eq (22)
    
    #     if a in self.continuous_cols:
    #         # Eq (22.b)
    #         res = 0
    #         for c in range(len(np.unique(self.data[:, -1]))):
    #             p_acx = self.__interpolated_prob(x, a, c)
    #             p_acy = self.__interpolated_prob(y, a, c)
    #             res += np.abs(p_acx - p_acy) ** 2
    #     elif a in self.discrete_cols:
    #         # VDM --> paper page 6
    #         res = self.__vdm(x, y, a) # Eq (22.a)
        
    #     return res
    
    
    
    # def __vdm(self, x: int, y: int, a: int): # Eq (8)
    #     res = 0
    #     q = 2 # It could also take value 1 TODO: decide which of the two values to use
        
    #     for c in range(len(np.unique(self.data[:, -1]))): 
    #         # num. instances in training set that have value x for attribute a
    #         N_ax = np.sum(self.data[:, a] == x)
    #         # num. of instances in the training set that have value x for attribute a and output class c
    #         N_axc = np.sum((self.data[:, a] == x) & (self.data[:, -1] == c))
            
    #         N_ay = np.sum(self.data[:, a] == y)
    #         N_ayc = np.sum((self.data[:, a] == y) & (self.data[:, -1] == c))

    #         P_axc = N_axc / N_ax # conditional probability that the output class is "c" given that attribute "a" has value "x"
    #         P_ayc = N_ayc / N_ay # conditional probability that the output class is "c" given that attribute "a" has value "y"
            
    #         res += np.abs(P_axc - P_ayc) ** q
            
    #     return res
    
    
    # def __interpolated_prob(self, x: float, a: int, c: int): # Eq (23)
    #     res = 0
        
    #     u = self.__discretize(x, a) # Determining the discrete value for attribute "a" correspodning to value "x".
        
    #     mid_au = self.__mid_point(u, a)
    #     if x < mid_au:
    #         # "The value of u is found by first setting u = discretize_a(x), and then substracting 1 from u if x < mid_au" 
    #         u -= 1
    #         mid_au = self.__mid_point(u, a)
        
    #     mid_au1 = self.__mid_point(u + 1, a)
        
    #     P_auc = self.P_avc[f"{a}_{u}_{c}"]
    #     P_au1c = self.P_avc[f"{a}_{u + 1}_{c}"] # Pray that this entry exists
        
    #     res = P_auc + ((x - mid_au) / (mid_au1 - mid_au)) * (P_au1c - P_auc)
        
    #     return res

    
    def __build_cache(self):
        """
        Precompute arrays and mappings to avoid repeated Python loops during distance computation.
        Builds:
          - self.num_classes, self.classes
          - self.s (number of discretization bins)
          - Per-attribute mins, maxs, widths
          - self.P_avc_arr[a] -> numpy array shape (s+2, n_classes) with P(a,u,c)
          - self.continuous_Pac[a] -> for continuous a: shape (num_instances, n_classes) with p_ac for each dataset value
          - self.discrete_Pvc[a] -> for discrete a: shape (num_instances, n_classes) with P_vc for each dataset value
          - self.discrete_value_Pvc[a] -> dict mapping value -> P_vc vector for quick lookup
        """
        data = self.data
        self.classes = np.unique(data[:, -1])
        self.num_classes = len(self.classes)
        self.s = max(5, self.num_classes)
        _, num_attributes = data.shape

        # Per-attribute stats
        self._min = np.min(data[:, :-1], axis=0)
        self._max = np.max(data[:, :-1], axis=0)
        
        self._width = (self._max - self._min) / float(self.s)
        

        # Build P_avc_arr for faster lookup (no need to take into consideration the output class)
        self.P_avc_arr = {}
        for a in range(num_attributes - 1):
            arr = np.zeros((self.s + 2, self.num_classes), dtype=float)
            for u in range(self.s + 2):
                for class_num, c in enumerate(self.classes):
                    arr[u, class_num] = self.P_avc[int(a), int(u), int(c)]
            self.P_avc_arr[a] = arr

        # Precompute per-point P_ac arrays for continuous attributes
        self.continuous_Pac = {}
        for a in self.continuous_cols:
            col = data[:, a].astype(float)
            self.continuous_Pac[a] = self._interpolated_prob_array(col, a)

        # Precompute per-point P_vc for discrete attributes and mapping value->P_vc
        self.discrete_Pvc = {}
        self.discrete_value_Pvc = {}
        for a in self.discrete_cols:
            col = data[:, a]
            unique_vals = np.unique(col)
            
            val_map = {}
            for v in unique_vals:
                Pvc = np.zeros(self.num_classes, dtype=float)
                
                for class_num, c in enumerate(self.classes):
                    Pvc[class_num] = self.P_avc[int(a), int(v), int(c)]
                val_map[v] = Pvc
            
            self.discrete_value_Pvc[a] = val_map
            
            # Per-point array
            per_point = np.vstack([val_map[val] for val in col])
            
            self.discrete_Pvc[a] = per_point

    def _interpolated_prob_array(self, xarr: np.ndarray, a: int): # Eq (23)
        """
        Vectorized version of __interpolated_prob for an array of x values.
        Returns array shape (len(xarr), n_classes).
        """
        x = xarr.astype(float)
        min_a = self._min[a]
        width_a = self._width[a]
        s = self.s

        # discretize vectorized: if x == max -> u = s, else floor((x-min)/w)+1
        u = np.floor((x - min_a) / width_a).astype(int) + 1

        # Compute midpoints
        mid_u = min_a + (width_a * u) + (width_a * 0.5)

        mask = x < mid_u
        u_adj = u.copy()
        u_adj[mask] = u_adj[mask] - 1
        u_adj = np.clip(u_adj, 0, s + 1)

        mid_au = min_a + (width_a * u_adj) + (width_a * 0.5)
        mid_au1 = min_a + (width_a * (u_adj + 1)) + (width_a * 0.5)

        # Lookup P arrays
        p_arr = self.P_avc_arr[a]
        P_auc = p_arr[u_adj]
        P_au1c = p_arr[np.clip(u_adj + 1, 0, s + 1)]

        # Linear interpolation
        denom = (mid_au1 - mid_au)
        factor = ((x - mid_au) / denom).reshape(-1, 1)
        res = P_auc + factor * (P_au1c - P_auc)

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
            
    
    # def __mid_point(self, u: int, a: int): # Eq (24)
    #     s = max(5, len(np.unique(self.data[:, -1])))
    #     width_a = (max(self.data[:, a]) - min(self.data[:, a])) / s  # eq 17

    #     return min(self.data[:, a]) + (width_a * u) + (width_a * 0.5)

    
    
    def __learn_P(self): # figure 5
        # Computing this matrix can be done independently for each attribute. For this reason, it is a function that is prone to parallelization.
        # Given that it is a time-consuming computation, especially for large datasets, using several processes can significantly decrease the computation time.
        def process_attribute(a):
            n_avc = defaultdict(int)
            n_av = defaultdict(int)
            p_avc_local = {}

            v_values_attribute = []
            for instance in self.data:
                x = instance[a]
                v = self.__discretize(x, a)
                v_values_attribute.append(v)
                c = instance[-1]
                n_avc[f"{a}_{v}_{c}"] += 1
                n_av[f"{a}_{v}"] += 1

            for v in np.unique(v_values_attribute):
                for c in np.unique(self.data[:, -1]):
                    if n_av[f"{a}_{v}"] == 0:
                        p_avc_local[f"{a}_{v}_{c}"] = 0
                    else:
                        p_avc_local[f"{a}_{v}_{c}"] = n_avc[f"{a}_{v}_{c}"] / n_av[f"{a}_{v}"]

            return p_avc_local

        # Comptue the results for each attribute in a parallel way 
        results = Parallel(n_jobs=-1, backend="loky")(delayed(process_attribute)(a)
                                                    for a in range(self.data.shape[1] - 1))

        # Combine the results from each parallel execution
        p_avc = defaultdict(float)
        for d in results:
            p_avc.update(d)
        
        max_v = max([float(elem.split("_")[1]) for elem in list(p_avc.keys())])
        
        matrix = np.zeros(shape=(int(self.data.shape[1]), int(max_v) + 2, len(np.unique(self.data[:, -1]))))
        
        
        for k, value in p_avc.items():
            elems = k.split("_")
            a = int(float(elems[0]))
            v = int(float(elems[1]))
            c = int(float(elems[2]))
            matrix[a, v, c] = value
        return matrix
        

        
            
    

class HEOM:
    # TODO: complete if the metric wants to be implemented
    pass

class GWHSM:
    # TODO: complete if the metric wants to be implemented
    pass