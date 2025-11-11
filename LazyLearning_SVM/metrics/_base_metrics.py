import numpy as np

__all__ = ["BaseMetrics"]

class BaseMetrics:
    @staticmethod
    def euclidean_dist(target: np.ndarray, instance: np.ndarray) -> np.ndarray:
        
        assert instance.ndim == 1, f"Instance should be a 1-dimensional array. Got dimensions {instance.shape}"
        
        instance_repeated = np.tile(instance, (target.shape[0], 1))
        euclidean_dist = np.linalg.norm(instance_repeated[:, :-1] - target[:, :-1], axis=1)
        
        sorted_idx = np.argsort(euclidean_dist)
        
        return target[sorted_idx]
    
    @staticmethod
    def cosine_dist(target: np.ndarray, instance: np.ndarray) -> np.ndarray:
        
        assert instance.ndim == 1, f"Instance should be a 1-dimensional array. Got dimensions {instance.shape}"
        
        instance_repeated = np.tile(instance, (target.shape[0], 1))
        cosine_dist = 1 - (np.einsum('ij,ij->i', instance_repeated[:, :-1], target[:, :-1]) / (np.linalg.norm(instance_repeated[:, :-1], axis=1) * np.linalg.norm(target[:, :-1], axis=1)))

        sorted_idx = np.argsort(cosine_dist)
        

        return target[sorted_idx]

