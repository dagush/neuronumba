import numpy as np
from scipy.special import erfcinv

# Matlab's corr2 function. Code taken from
# https://stackoverflow.com/questions/29481518/python-equivalent-of-matlab-corr2
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y


def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r


# Matlab's corr function. Based on the code from
# https://stackoverflow.com/questions/71563937/pandas-autocorr-returning-different-calcs-then-my-own-autocorrelation-function
def corr(A, B):
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    B = (B - B.mean(axis=0)) / B.std(axis=0)
    correlation = (np.dot(B.T, A) / B.shape[0]).T
    return correlation


# Matlab's autocorr function. From
# https://stackoverflow.com/questions/61624985/python-use-of-corrcoeff-to-achieve-matlabs-corr-function
def autocorr(x, lags):
    autocorrs = np.ones(lags+1)  # just to initialize autocorr[0] = 1 ;-)
    for lag in range(1, lags+1):
        series = x[lag:]
        series_auto = x[:-lag]
        corr = 0
        var_x1 = 0
        var_x2 = 0
        for j in range(len(series)):
            x1 = series[j] - np.average(series)
            x2 = series_auto[j] - np.average(series_auto)
            corr += x1*x2
            var_x1  += x1**2
            var_x2 += x2**2
        autocorrs[lag] = corr/((var_x1*var_x2) ** 0.5)
    return autocorrs


# Computes the correlation matrix from a covariance matrix. Replaces
# MATLAB's corrcov function. From
# https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
def correlation_from_covariance(covariance):
    """
    Parameters
    ----------
    covariance : matrix with covariance values, format (n_roi, n_roi)

    Returns
    -------
    correlation : matrix with correlation values, format (n_roi, n_roi)

    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# Rejects outliers based on the median absolute deviations (MAD). Replaces
# MATLAB's rmoutliers function:
# https://uk.mathworks.com/help/matlab/ref/rmoutliers.html
# Code adapted from:
# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
# and with modifications explained here:
# https://uk.mathworks.com/help/matlab/ref/filloutliers.html#bvml247
def reject_outliers(data, m = 3.):
    c = -1/(np.sqrt(2)*erfcinv(3/2))
    array_data = np.array(data)
    d = np.abs(array_data - np.median(data))
    mdev = c*np.median(d)
    s = d / (mdev if mdev else 1.)
    return array_data[s < m]