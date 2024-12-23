from scipy import linalg

from neuronumba.observables.linear.base_linear import ObservableLinear
from neuronumba.tools.matlab_tricks import correlation_from_covariance


class LinearFC(ObservableLinear):
    def from_matrix(self, A, Qn):
        N = int(A.shape[0] / 2)
        # Solves the Lyapunov equation: A*X + X*Ah = Q, with Ah the conjugate transpose of A
        CVth = linalg.solve_continuous_lyapunov(A, -Qn)
        # simulated FC
        FCth = correlation_from_covariance(CVth)
        # Functional connectivity matrix (FC)
        FC = FCth[0:N, 0:N]
        CV = CVth[0:N, 0:N]

        return {'FC': FC, 'CVth': CVth, 'CV': CV}