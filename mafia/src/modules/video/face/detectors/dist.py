import numpy as np

def distance(p: np.ndarray, q: np.ndarray):
    return np.sqrt(((p - q) ** 2).sum())
