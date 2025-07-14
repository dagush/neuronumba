from abc import ABC, abstractmethod
import numpy as np

class DistanceRule(ABC):
    @abstractmethod
    def compute(self, x_i, x_j, lambda_val):
        pass