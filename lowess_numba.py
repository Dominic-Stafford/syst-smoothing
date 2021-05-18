"""Module provides functions for LOWESS smoothing [1].

[1] https://en.wikipedia.org/wiki/Local_regression
"""


import numpy as np
from numba import njit

@njit(cache=True)
def lowess(x, y, bandwidth, weights=None):
    """Smooth a function using LOWESS algorithm.
    
    Parameters:
        x:  Sorted array of x coordinates.
        y:  Corresponding values of the function.
        bandwidth: Bandwith that defines the size of the neighbourhood
            to be considered for each point.
        weights:  Additional weights for LOWESS fit.  They should
            normally be inversely proportional to squared uncertainties
            associated with each point.
    
    Return value:
        An array with results of the smoothing at the same x coordinates
        as given in inputs.
    
    At each point perform a linear regression.  Only points that fall
    inside a window with a half-size equal to the given bandwidth are
    considered.  The least-squares fit is performed using weights
    computed by the tricube function of the distance in x to the central
    point in the window, divided by the bandwith.
    
    This function is similar to implementation from statsmodels [1] but
    operates with a fixed bandwith instead of a fixed number of
    neighbours.
    
    [1] http://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    """
    smooth_y = np.zeros(len(y))
    external_weights = np.asarray(weights) if weights is not None else np.ones(len(x))
    
    for i in range(len(y)):
        
        # Find points whose distance from x[i] is not greater than
        # bandwidth.  The last point is not included in the range.
        start = np.searchsorted(x, x[i] - bandwidth, side='left')
        end = np.searchsorted(x, x[i] + bandwidth, side='right')
        
        
        # Compute weights for selected points.  They are evaluated as
        # the tricube function of the distance, in units of bandwidth.
        # Negative weights are clipped.
        distances = np.abs(x[start:end] - x[i]) / bandwidth
        weights = (1 - distances ** 3) ** 3
        weights *= external_weights[start:end]
        weights[weights < 0.] = 0.
        
        # To simplify computation of mean values below, normalize
        # weights
        weights /= np.sum(weights)
        
        
        # Perform linear fit to selected points.  The range is centered
        # at x[i] so that only the constant term needs to be computed.
        # This also improves numerical stability.  The computation is
        # carried using an analytic formula.
        x_fit = x[start:end] - x[i]
        
        mean_x = np.dot(weights, x_fit)
        mean_y = np.dot(weights, y[start:end])
        mean_x2 = np.dot(weights, x_fit ** 2)
        mean_xy = np.dot(weights, x_fit * y[start:end])
        
        smooth_y[i] = (mean_x2 * mean_y - mean_x * mean_xy) / (mean_x2 - mean_x ** 2)
    
    return smooth_y
