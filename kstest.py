"""This module implements two-sided K-S tests. The 1-D version is from scipy, the 2-D version
is that originated by Peacock 1984: http://adsabs.harvard.edu/abs/1983MNRAS.202..615P
and Fasano 1987 http://adsabs.harvard.edu/abs/1987MNRAS.225..155F

See also NR 14.8 (although they are a bit vague). Note that this uses NR results,
but not their code, so is ok for distribution.
"""

import scipy.stats as st
import numpy as np
import math

def ks_2samp(data1, data2):
    """Computes the Kolmogorov-Smirnof statistic on 2 samples.
       This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
        Parameters :
        a, b : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous distribution, sample sizes can be different

        Returns :
        D : float   KS statistic
        p-value : float  two-tailed p-value: high value means
        we cannot reject the hypothesis that numbers are from the same distribution
        ie, low D => high p
    """
    return st.ks_2samp(data1, data2)

def ks_2d_2samp(data1, data2):
    """Computes the 2-dimensional Kolmogorov-Smirnof statistic on 2 samples.
       This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
        Parameters :
        a, b : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous distribution, sample sizes can be different

        Returns :
        D : float   KS statistic
        p-value : float  two-tailed p-value.
        High value means we cannot reject the hypothesis that they are from the same distribution.
        low D => high p
    """
    #Compute D using data1 as the origins
    D1 = np.max([max_diff_for_orig(dd, data1, data2) for dd in data1])
    #Compute D using data2 as the origins
    D2 = np.max([max_diff_for_orig(dd, data1, data2) for dd in data2])
    #Their mean
    D = (D1+D2)/2.
    #The approximate p-value: this is detailed in NR 14.8
#     neff = npt1*npt2/(1.*npt1+npt2)
#     (rr1,p) = st.pearsonr(data1[:,0], data1[:,1])
#     (rr2,p) = st.pearsonr(data2[:,0], data2[:,1])
#     reff = (rr1**2+rr2**2)/2.
#     ksarg = neff*D/(1+np.sqrt(1-reff)*(025-0.75/np.sqrt(neff)))
#     pval = ksdist(ksarg)
    return D #(D, pval)

def max_diff_for_orig(orig, data1, data2):
    """For a given origin, orig, compute the fraction of each data point in each quadrant,
    then compute the maximum distance between the distributions, and return it."""
    count1 = count_quadrant(orig, data1)
    count2 = count_quadrant(orig, data2)
    return np.max(np.abs(count1-count2))

def count_quadrant(orig, xx):
    """Count fraction of points in each quadrant around an origin orig = (x,y), from an nd-array of points.
    Returns:
        count: array of length 4.
        Each element is the fraction of items in each quadrant, measured clockwise from x > x_i, y > y_i in quadrant 0.
    """
    count = np.zeros(4)
    #The clockwise arrangement of the = digns mean that (0,0) is in no quadrant
    # x > 0, y >= 0
    count[0] = np.size(np.where(np.logical_and(xx[:,0] > orig[0], xx[:,1] >= orig[1])))
    # x >= 0, y < 0
    count[1] = np.size(np.where(np.logical_and(xx[:,0] >= orig[0], xx[:,1] < orig[1])))
    # x < 0, y <= 0
    count[2] = np.size(np.where(np.logical_and(xx[:,0] < orig[0], xx[:,1] <= orig[1])))
    # x <= 0, y > 0
    count[3] = np.size(np.where(np.logical_and(xx[:,0] <= orig[0], xx[:,1] > orig[1])))
    #One element (the origin) will be in no quadrant
    nn = np.shape(xx)[0]-1
    count/=(1.*nn)
    return count

def ksdist(x):
    """The K-S distribution, from the description in NR.
    Scipy seems to do this too, but I don't understand their documentation,
    so I implement it myself.
    Defined st ksdist(0) == 1 and ksdist(infty) == 0.
    """
    if x == 0:
        return 1
    if x > 1.18:
        return _ksdist1(x)
    else:
        return _ksdist2(x)

def _ksdist1(x):
    """Helper for above. Series is eq. 6.14.56 of NR.
    Best for when x > 1.18"""
    total = 0
    #Should be < 3 iterations in practice
    for j in range(20):
        extra = (-1)**j*np.exp(-2*(j+1)**2*x**2)
        total += extra
        if extra == 0 or extra/total < 1e-7:
            break
    total*=2
    return total

def _ksdist2(x):
    """Helper for above. Series is eq. 6.14.57 of NR.
    Best for when x < 1.18"""
    total = 0
    #Should be < 3 iterations in practice
    for j in range(20):
        extra = np.exp(-(2*j-1)**2*math.pi**2/(8*x**2))
        total += extra
        if extra == 0 or extra/total < 1e-7:
            break
    total*=np.sqrt(2*math.pi)/x
    return 1-total
