import numpy as np
__all__ = ["MCNN"]


class MCNN:
    
    @staticmethod
    def _get_class_data(data: np.ndarray):
        class_data = {}
        if data.size == 0:
            return {}, []
        
        labels = np.unique(data[:, -1])
        for label in labels:
            class_data[label] = data[data[:, -1] == label]
        return class_data, labels 
    
    @staticmethod
    def _get_centroid(data: np.ndarray):
        if data.size == 0:
            return None
        return np.mean(data[:, :-1], axis=0)
    
    @staticmethod
    def _nearest_neighbor_label(sample: np.ndarray, prototypes: np.ndarray):
        if prototypes.size == 0 or prototypes.ndim != 2:
            return None 
        sample_features = sample[:-1]
        prototype_features = prototypes[:, :-1]
        distances = np.sum(np.square(prototype_features - sample_features), axis=1)
        nearest_index = np.argmin(distances)
        return prototypes[nearest_index, -1]
    
    @staticmethod
    def _find_closest_sample_to_centroid(data: np.ndarray, centroid: np.ndarray):
        if centroid is None or data.size == 0:
            return None
        features = data[:, :-1]
        distances = np.linalg.norm(features - centroid, axis=1)
        closest_index = np.argmin(distances)
        return data[closest_index]

    @staticmethod
    def reduce(data: np.ndarray) -> np.ndarray:
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
                class_data_s_new, labels = MCNN._get_class_data(S_new) 
                new_prototypes = []
                for label in labels:
                    data_j = class_data_s_new[label]
                    
                    if data_j.size > 0:
                        centroid_j = MCNN._get_centroid(data_j) 
                        prototype_j = MCNN._find_closest_sample_to_centroid(data_j, centroid_j)
                        
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
                    nn_label = MCNN._nearest_neighbor_label(sample, P_current) 
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
                nn_label = MCNN._nearest_neighbor_label(sample, Q)
                if nn_label != sample[-1]:
                    S_misclassified_final = np.vstack([S_misclassified_final, sample])
            S_current = S_misclassified_final
            
            if S_misclassified_final.size == 0:
                return Q