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
    smoothY = np.zeros(len(y))
    
    for i in range(len(y)):
        
        # Find points whose distance from x[i] is not greater than
        # bandwidth.  The last point is not included in the range.
        start = np.searchsorted(x, x[i] - bandwidth, side='left')
        end = np.searchsorted(x, x[i] + bandwidth, side='right')
        
        
        # Compute weights for selected points.  They are evaluated as
        # the tricube function of the distance, in units of bandwidth.
        # Negative weights are clipped.
        distances = np.abs(x[start:end] - x[i]) / bandwidth
        weights = (1 - distances**3)**3
        weights[weights < 0.] = 0.
        
        # To simplify computation of mean values below, normalize
        # weights
        weights /= np.sum(weights)
        
        
        # Perform linear fit to selected points.  The range is centered
        # at x[i] so that only the constant term needs to be computed.
        # This also improves numerical stability.  The computation is
        # carried using an analytic formula.
        xFit = x[start:end] - x[i]
        
        meanX = np.dot(weights, xFit)
        meanY = np.dot(weights, y[start:end])
        meanX2 = np.dot(weights, xFit**2)
        meanXY = np.dot(weights, xFit * y[start:end])
        
        smoothY[i] = (meanX2 * meanY - meanX * meanXY) / (meanX2 - meanX**2)
    
    return smoothY


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
    
    
    ySmooth_np = np.empty_like(y_np)
    cdef float64_t [:] ySmooth = ySmooth_np
    
    cdef:
        Py_ssize_t nPoints = len(y)
        Py_ssize_t iCentral, iCurrent, d
        float64_t xTrans[2]
        float64_t distance2, weight, det
        float64_t sumW, sumWX0X1, sumWY
        float64_t sumWX[2]
        float64_t sumWXQ[2]
        float64_t sumWXY[2]
        float64_t m[3][3]
    
    
    # Loop over given points
    for iCentral in range(nPoints):
        
        sumW = sumWX0X1 = sumWY = 0.
        
        for d in range(2):
            sumWX[d] = 0.
            sumWXQ[d] = 0.
            sumWXY[d] = 0.
        
        
        # Compute various weighted sums looping over all points
        for iCurrent in range(nPoints):
            
            # Centre x coordinates at the current point and express them
            # in units of bandwidths along each axis.  The centring will
            # allow to compute only one of the three parameters defining
            # a plane.
            for d in range(2):
                xTrans[d] = (x_v[iCurrent, d] - x_v[iCentral, d]) / h[d]
            
            distance2 = xTrans[0]**2 + xTrans[1]**2
            
            if distance2 > 1:
                continue
            
            
            weight = (1 - sqrt(distance2)**3)**3
            
            sumW += weight
            sumWX0X1 += weight * xTrans[0] * xTrans[1]
            sumWY += weight * y_v[iCurrent]
            
            for d in range(2):
                sumWX[d] += weight * xTrans[d]
                sumWXQ[d] += weight * xTrans[d]**2
                sumWXY[d] += weight * xTrans[d] * y_v[iCurrent]
        
        
        # Set smoothed y at the current point to the constant term in
        # the equation for the plane that fits the data in the local
        # neighbourhood.  Use weighted least squares fit.  The resulting
        # system of linear equations for parameters of the plane is
        # solved using Cramer's rule.
        m[0][0] = 1.
        m[1][1] = sumWXQ[0] / sumW
        m[2][2] = sumWXQ[1] / sumW
        m[0][1] = m[1][0] = sumWX[0] / sumW
        m[0][2] = m[2][0] = sumWX[1] / sumW
        m[1][2] = m[2][1] = sumWX0X1 / sumW
        
        det = determinant_3x3(m)
        
        m[0][0] = sumWY / sumW
        m[1][0] = sumWXY[0] / sumW
        m[2][0] = sumWXY[1] / sumW
        
        ySmooth[iCentral] = determinant_3x3(m) / det
    
    
    return ySmooth_np



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
    
    
    ySmooth_np = np.empty_like(y_np)
    cdef float64_t [:,:] ySmooth = ySmooth_np
    
    cdef Py_ssize_t n[2]
    n[0] = y_np.shape[0]
    n[1] = y_np.shape[1]
    
    cdef:
        Py_ssize_t iCentral[2]
        Py_ssize_t iStart[2]
        Py_ssize_t iEnd[2]
        Py_ssize_t iCurrent[2]
        int d
        
        float64_t xTrans[2]
        float64_t distance2, weight, det
        
        float64_t sumW, sumWX0X1, sumWY
        float64_t sumWX[2]
        float64_t sumWXQ[2]
        float64_t sumWXY[2]
        float64_t m[3][3]
    
    
    for iCentral[0] in range(n[0]):
        
        # The window along the first coordinate around the current
        # point.  Points with end indices are not included.
        iStart[0] = upper_bound(x0_v, 0, iCentral[0], x0_v[iCentral[0]] - h[0])
        iEnd[0] = upper_bound(x0_v, iCentral[0], n[0], x0_v[iCentral[0]] + h[0])
        
        for iCentral[1] in range(n[1]):
            
            # The window along the second coordinate
            iStart[1] = upper_bound(x1_v, 0, iCentral[1], x1_v[iCentral[1]] - h[1])
            iEnd[1] = upper_bound(x1_v, iCentral[1], n[1], x1_v[iCentral[1]] + h[1])
            
            
            sumW = sumWX0X1 = sumWY = 0.
            
            for d in range(2):
                sumWX[d] = 0.
                sumWXQ[d] = 0.
                sumWXY[d] = 0.
            
            # Loop over points included in the 2D window and compute
            # weighted sums
            for iCurrent[0] in range(iStart[0], iEnd[0]):
                for iCurrent[1] in range(iStart[1], iEnd[1]):
                    
                    # Centre x coordinates at the current point and
                    # express them in units of bandwidths along each
                    # axis.  The centring will allow to compute only one
                    # of the three parameters defining a plane.
                    xTrans[0] = (x0_v[iCurrent[0]] - x0_v[iCentral[0]]) / h[0]
                    xTrans[1] = (x1_v[iCurrent[1]] - x1_v[iCentral[1]]) / h[1]
                    
                    distance2 = xTrans[0]**2 + xTrans[1]**2
                    
                    if distance2 > 1:
                        continue
                    
                    
                    weight = (1 - sqrt(distance2)**3)**3
                    
                    sumW += weight
                    sumWX0X1 += weight * xTrans[0] * xTrans[1]
                    sumWY += weight * y_v[iCurrent[0], iCurrent[1]]
                    
                    for d in range(2):
                        sumWX[d] += weight * xTrans[d]
                        sumWXQ[d] += weight * xTrans[d]**2
                        sumWXY[d] += weight * xTrans[d] * y_v[iCurrent[0], iCurrent[1]]
            
            
            # Set smoothed y at the current point to the constant term
            # in the equation for the plane that fits the data in the
            # local neighbourhood.  Use weighted least squares fit.  The
            # resulting system of linear equations for parameters of the
            # plane is solved using Cramer's rule.
            m[0][0] = 1.
            m[1][1] = sumWXQ[0] / sumW
            m[2][2] = sumWXQ[1] / sumW
            m[0][1] = m[1][0] = sumWX[0] / sumW
            m[0][2] = m[2][0] = sumWX[1] / sumW
            m[1][2] = m[2][1] = sumWX0X1 / sumW
            
            det = determinant_3x3(m)
            
            m[0][0] = sumWY / sumW
            m[1][0] = sumWXY[0] / sumW
            m[2][0] = sumWXY[1] / sumW
            
            ySmooth[iCentral[0], iCentral[1]] = determinant_3x3(m) / det
    
    return ySmooth_np



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
