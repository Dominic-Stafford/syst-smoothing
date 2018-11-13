"""Module provides functions for LOWESS smoothing [1].

[1] https://en.wikipedia.org/wiki/Local_regression
"""


import numpy as np

from numpy cimport float64_t 

cimport cython
from libc.math cimport sqrt


def lowess(x, y, bandwidth):
    """Smooth a function using LOWESS algorithm.
    
    Parameters:
        x:  Sorted array of x coordinates.
        y:  Corresponding values of the function.
        bandwidth: Bandwith that defines the size of the neighbourhood
            to be considered for each point.
    
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
    
    x = np.asarray(x)
    smooth_y = np.zeros(len(y))
    
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lowess2d(x, y, bandwidth):
    """Smooth a function of 2D data using LOWESS algorithm.
    
    Arguments:
        x:  array_like of shape (n, 2)
            Sequence of n 2D points.  Does not need to be ordered.
        y:  array_like of length n
            Input values of the fuction evaluated at each point.
        bandwidth: array_like of length 2
            Bandwidths to be used for each coordinate.
    
    Return value:
        A NumPy array with smoothed values of the function.
    
    For each point, fit data in its neighbourhood with a plane,
    weighting down points that are far away.  The smoothed value is
    computed as the coordinate of the plane at that central point.
    """
    
    # Create memory views of input arrays.  Convert them to float64 if
    # needed.
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    
    if x_np.shape[1] != 2:
        raise RuntimeError('Size of array x along axis 1 must be 2.')
    
    if x_np.shape[0] != y_np.shape[0]:
        raise RuntimeError('Dimensions of arrays x and y do not match.')
    
    cdef float64_t [:, :] x_v = x_np
    cdef float64_t [:] y_v = y_np
    
    cdef float64_t h[2]
    h[0], h[1] = bandwidth
    
    
    y_smooth_np = np.empty_like(y_np)
    cdef float64_t [:] y_smooth = y_smooth_np
    
    cdef:
        Py_ssize_t num_points = len(y)
        Py_ssize_t i_central, i_current, d
        float64_t x_trans[2]
        float64_t distance2, weight, det
        float64_t sum_w, sum_wx0x1, sum_wy
        float64_t sum_wx[2]
        float64_t sum_wxq[2]  # q for squared x
        float64_t sum_wxy[2]
        float64_t m[3][3]
    
    
    # Loop over given points
    for i_central in range(num_points):
        
        sum_w = sum_wx0x1 = sum_wy = 0.
        
        for d in range(2):
            sum_wx[d] = 0.
            sum_wxq[d] = 0.
            sum_wxy[d] = 0.
        
        
        # Compute various weighted sums looping over all points
        for i_current in range(num_points):
            
            # Centre x coordinates at the current point and express them
            # in units of bandwidths along each axis.  The centring will
            # allow to compute only one of the three parameters defining
            # a plane.
            for d in range(2):
                x_trans[d] = (x_v[i_current, d] - x_v[i_central, d]) / h[d]
            
            distance2 = x_trans[0] ** 2 + x_trans[1] ** 2
            
            if distance2 > 1:
                continue
            
            
            weight = (1 - sqrt(distance2) ** 3) ** 3
            
            sum_w += weight
            sum_wx0x1 += weight * x_trans[0] * x_trans[1]
            sum_wy += weight * y_v[i_current]
            
            for d in range(2):
                sum_wx[d] += weight * x_trans[d]
                sum_wxq[d] += weight * x_trans[d] ** 2
                sum_wxy[d] += weight * x_trans[d] * y_v[i_current]
        
        
        # Set smoothed y at the current point to the constant term in
        # the equation for the plane that fits the data in the local
        # neighbourhood.  Use weighted least squares fit.  The resulting
        # system of linear equations for parameters of the plane is
        # solved using Cramer's rule.
        m[0][0] = 1.
        m[1][1] = sum_wxq[0] / sum_w
        m[2][2] = sum_wxq[1] / sum_w
        m[0][1] = m[1][0] = sum_wx[0] / sum_w
        m[0][2] = m[2][0] = sum_wx[1] / sum_w
        m[1][2] = m[2][1] = sum_wx0x1 / sum_w
        
        det = determinant_3x3(m)
        
        m[0][0] = sum_wy / sum_w
        m[1][0] = sum_wxy[0] / sum_w
        m[2][0] = sum_wxy[1] / sum_w
        
        y_smooth[i_central] = determinant_3x3(m) / det
    
    
    return y_smooth_np



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lowess2d_grid(x0, x1, y, bandwidth):
    """Smooth a function of 2D data using LOWESS algorithm.
    
    Input values for the function are given on a grid.
    
    Arguments:
        x0, x1:  array_like
            Sorted arrays that define the grid along the two axes.
        y:  array_like
            2D array with input values of the fuction at each point on
            the grid.  Axis 0 corresponds to x0, axis 1 to x1.
        bandwidth: array_like of length 2
            Bandwidths to be used for each coordinate.
    
    Return value:
        A NumPy array with smoothed values of the function.
    
    For each point, fit data in its neighbourhood with a plane,
    weighting down points that are far away.  The smoothed value is
    computed as the coordinate of the plane at that central point.
    """
    
    # Create memory views of input arrays.  Convert them to float64 if
    # needed.
    x0_np = np.asarray(x0, dtype=np.float64)
    x1_np = np.asarray(x1, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    
    if y_np.shape[0] != x0_np.shape[0] or y_np.shape[1] != x1_np.shape[0]:
        raise RuntimeError('Dimenstions of array y do not match those of x0 and x1')
    
    cdef float64_t [:] x0_v = x0_np
    cdef float64_t [:] x1_v = x1_np
    cdef float64_t [:, :] y_v = y_np
    
    cdef float64_t h[2]
    h[0] = bandwidth[0]
    h[1] = bandwidth[1]
    
    
    y_smooth_np = np.empty_like(y_np)
    cdef float64_t [:,:] y_smooth = y_smooth_np
    
    cdef Py_ssize_t n[2]
    n[0] = y_np.shape[0]
    n[1] = y_np.shape[1]
    
    cdef:
        Py_ssize_t i_central[2]
        Py_ssize_t i_start[2]
        Py_ssize_t i_end[2]
        Py_ssize_t i_current[2]
        int d
        
        float64_t x_trans[2]
        float64_t distance2, weight, det
        
        float64_t sum_w, sum_wx0x1, sum_wy
        float64_t sum_wx[2]
        float64_t sum_wxq[2]  # q for squared x
        float64_t sum_wxy[2]
        float64_t m[3][3]
    
    
    for i_central[0] in range(n[0]):
        
        # The window along the first coordinate around the current
        # point.  Points with end indices are not included.
        i_start[0] = upper_bound(x0_v, 0, i_central[0], x0_v[i_central[0]] - h[0])
        i_end[0] = upper_bound(x0_v, i_central[0], n[0], x0_v[i_central[0]] + h[0])
        
        for i_central[1] in range(n[1]):
            
            # The window along the second coordinate
            i_start[1] = upper_bound(x1_v, 0, i_central[1], x1_v[i_central[1]] - h[1])
            i_end[1] = upper_bound(x1_v, i_central[1], n[1], x1_v[i_central[1]] + h[1])
            
            
            sum_w = sum_wx0x1 = sum_wy = 0.
            
            for d in range(2):
                sum_wx[d] = 0.
                sum_wxq[d] = 0.
                sum_wxy[d] = 0.
            
            # Loop over points included in the 2D window and compute
            # weighted sums
            for i_current[0] in range(i_start[0], i_end[0]):
                for i_current[1] in range(i_start[1], i_end[1]):
                    
                    # Centre x coordinates at the current point and
                    # express them in units of bandwidths along each
                    # axis.  The centring will allow to compute only one
                    # of the three parameters defining a plane.
                    x_trans[0] = (x0_v[i_current[0]] - x0_v[i_central[0]]) / h[0]
                    x_trans[1] = (x1_v[i_current[1]] - x1_v[i_central[1]]) / h[1]
                    
                    distance2 = x_trans[0] ** 2 + x_trans[1] ** 2
                    
                    if distance2 > 1:
                        continue
                    
                    
                    weight = (1 - sqrt(distance2) ** 3) ** 3
                    
                    sum_w += weight
                    sum_wx0x1 += weight * x_trans[0] * x_trans[1]
                    sum_wy += weight * y_v[i_current[0], i_current[1]]
                    
                    for d in range(2):
                        sum_wx[d] += weight * x_trans[d]
                        sum_wxq[d] += weight * x_trans[d] ** 2
                        sum_wxy[d] += weight * x_trans[d] * y_v[i_current[0], i_current[1]]
            
            
            # Set smoothed y at the current point to the constant term
            # in the equation for the plane that fits the data in the
            # local neighbourhood.  Use weighted least squares fit.  The
            # resulting system of linear equations for parameters of the
            # plane is solved using Cramer's rule.
            m[0][0] = 1.
            m[1][1] = sum_wxq[0] / sum_w
            m[2][2] = sum_wxq[1] / sum_w
            m[0][1] = m[1][0] = sum_wx[0] / sum_w
            m[0][2] = m[2][0] = sum_wx[1] / sum_w
            m[1][2] = m[2][1] = sum_wx0x1 / sum_w
            
            det = determinant_3x3(m)
            
            m[0][0] = sum_wy / sum_w
            m[1][0] = sum_wxy[0] / sum_w
            m[2][0] = sum_wxy[1] / sum_w
            
            y_smooth[i_central[0], i_central[1]] = determinant_3x3(m) / det
    
    return y_smooth_np



cdef float64_t determinant_3x3(float64_t m[3][3]):
    """Compute determinant of 3x3 matrix using explicit formula."""
    
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - \
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + \
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int upper_bound(float64_t [:] array, int start, int end, float64_t x):
    """Find index of the first element that is bigger than x.
    
    Arguments:
        array:  Sorted array.
        start, end:  Indices that define a subrange of the array to be
            considered.  Index 'end' is not included in the range.
        x:  Value to compare against.
    
    Return value:
        Index of the first element of the array, within the given
        subrange, that is strictly bigger than x.  If x is bigger or
        equal to the last element in the subrange, 'end' is returned.
    
    Implemented with a binary search.
    """
    
    if x < array[start]:
        return start
    elif x >= array[end - 1]:
        return end
    
    cdef int middle
    
    while end - start > 1:
        
        middle = start + (end - start) // 2
        
        if array[middle] > x:
            end = middle
        else:
            start = middle
    
    return end
