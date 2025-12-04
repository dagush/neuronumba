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
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, x_i, x_j, i , j):
        pass


class ExponentialDistanceRule(DistanceRule):
    def __init__(self, lambda_val=0.18):
        self.lambda_val = lambda_val

    def compute(self, x, y, i, j):
        """
        Compute the exponential distance between two vectors.
        """
        dist = np.linalg.norm(np.array(x) - np.array(y))
        return np.exp(-self.lambda_val * dist)



class EDRLongDistance(DistanceRule):
    """
    Exponential Distance Rule amb connexions long-range afegides,
    només si la distància entre nodes > 3 * lambda.
    """

    def __init__(self, lambda_val: float = 0.18, sc = None):
        super().__init__()
        self.lambda_val = lambda_val
        self.sc = sc

    def compute(self, x_i, x_j, i, j):

        dist = distance.euclidean(x_i, x_j)
        weight = np.exp(-self.lambda_val * dist)

        # Si la distància és > 3 * lambda, pot tenir connexió long-range
        if self.sc[i,j] > 3 * weight:
                weight = self.sc[i,j]

        return weight


