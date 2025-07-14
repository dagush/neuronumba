import numpy as np
from scipy import signal

from neuronumba.basic.attr import Attr
from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.tools import matlab_tricks
from distance_rule import DistanceRule

class EdrLongRangeDistanceRule(DistanceRule):
    def __init__(self, edr_threshold=0.5):
        self.edr_threshold = edr_threshold

    def edr(self, x, y):
        count = 0
        for xi, yi in zip(x, y):
            if abs(xi - yi) > self.edr_threshold:
                count += 1
        return count / max(len(x), len(y))

    def ld(self, x, y):
        dp = np.zeros((len(x)+1, len(y)+1), dtype=int)
        for i in range(len(x)+1):
            dp[i][0] = i
        for j in range(len(y)+1):
            dp[0][j] = j

        for i in range(1, len(x)+1):
            for j in range(1, len(y)+1):
                cost = 0 if x[i-1] == y[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[len(x)][len(y)]

    def compute(self, x, y):
        edr_dist = self.edr(x, y)
        ld_dist = self.ld(x, y)
        return edr_dist + ld_dist