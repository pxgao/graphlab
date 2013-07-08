"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from numpy.ma.core import exp, shape, reshape, sqrt
from numpy.ma.extras import median
from scipy.spatial.distance import squareform, pdist, cdist

class GaussianKernel(object):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d array, samples on right hand side
        Y - 2d array, samples on left hand side, can be None in which case
            it is replaced by X
        """
        
        # bring to 2d array form if 1d
        assert(len(shape(X))==2)
            
        if Y is not None:
            assert(len(shape(X))==2)
                
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            sq_dists = cdist(X, Y, 'sqeuclidean')
    
        K = exp(-0.5 * (sq_dists) / self.sigma ** 2)
        return K
    
    @staticmethod
    def get_sigma_median_heuristic(X):
        dists=squareform(pdist(X, 'euclidean'))
        median_dist=median(dists[dists>0])
        sigma=sqrt(0.5*median_dist)
        return sigma
