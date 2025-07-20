from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import distance



class DistanceRule(ABC):
    """
        Long range connections framework, from:
        Gustavo Deco, Yonathan Sanz Perl,Peter Vuust,Enzo Tagliazucchi,Henry Kennedy,Morten L. Kringelbach. Rare long-range cortical connections enhance human information processing,
        Current Biology (Volume 31, Issue 20, 2021,pp. 4436-4448.E5)
        https://doi.org/10.1016/j.cub.2021.07.064
        (https://www.sciencedirect.com/science/article/pii/S096098222101054X)

        Code by Gustavo Deco, 2021.
        Translated by Lisa Haz, July 2, 2024
        """
    @abstractmethod
    def compute(self, x_i, x_j, lambda_val):
        pass


class ExponentialDistanceRule(DistanceRule):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute(self, x, y, lambda_val):
        """
        Compute the exponential distance between two vectors.
        """
        dist = np.linalg.norm(np.array(x) - np.array(y))
        return np.exp(-lambda_val * dist)



class EDRLongDistance(DistanceRule):
    """
    Exponential Distance Rule amb connexions long-range afegides,
    només si la distància entre nodes > 3 * lambda.
    """

    def __init__(self, lambda_val: float = 0.18, lr_fraction: float = 0.05, seed: int = 42):
        super().__init__()
        self.lambda_val = lambda_val
        self.lr_fraction = lr_fraction
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.weights = None  # Es generarà només un cop

    def compute(self, x_i, x_j, lambda_val):

        dist = distance.euclidean(x_i, x_j)
        weight = np.exp(-lambda_val * dist)

        # Si la distància és > 3 * lambda, pot tenir connexió long-range
        if dist > 3 * lambda_val:
            # Decidir aleatòriament si hi ha connexió long-range segons lr_fraction (probabilitat)
            if self.rng.random() < self.lr_fraction:
                weight = 1.0

        return weight


