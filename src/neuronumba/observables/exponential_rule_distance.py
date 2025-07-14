import numpy as np
from neuronumba.observables.distance_rule import DistanceRule

class ExponentialDistanceRule(DistanceRule):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute(self, x, y, lambda_val):
        """
        Compute the exponential distance between two vectors.
        """
        dist = np.linalg.norm(np.array(x) - np.array(y))
        return np.exp(-lambda_val * dist)

