"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from numpy.core.shape_base import vstack
from numpy.linalg.linalg import solve
from numpy.ma.core import zeros, ones, sqrt

def incomplete_cholesky(X, kernel, eta, power=1, blocksize=100):
    """
    Computes the incomplete Cholesky factorisation of the kernel matrix defined
    by samples X and a given kernel. The kernel is evaluated on-the-fly.
    The optional power parameter is used to multiply the kernel output with
    itself.
    
    Original code from "Kernel Methods for Pattern Analysis" by Shawe-Taylor and
    Cristianini.
    Modified to compute kernel on the fly, to use kernels multiplied with 
    themselves (tensor product), and optimised speed via using vector
    operations and not pre-allocate full kernel matrix memory, but rather
    allocate memory of low-rank kernel block-wise
    Changes by Heiko Strathmann
    
    parameters:
    X         - list of input vectors to evaluate kernel on
    kernel    - a kernel object with a kernel method that takes 2d-arrays
                and returns a psd kernel matrix
    eta       - precision cutoff parameter for the low-rank approximation.
                Lies is (0,1) where smaller means more accurate.
    power     - every kernel evaluation is multiplied with itself this number
                of times. Zero is supported
    blocksize - tuning parameter for speed, determines how rows elements are
                allocated in a block for the (growing) kernel matrix. Larger
                means faster algorithm (to some extend if low rank dimension
                is larger than blocksize)
    
    output:
    K_chol, ell, I, R, W, where
    K    - is the kernel using only the pivot index features
    I    - is a list containing the pivots used to compute K_chol
    R    - is a low-rank factor such that R.T.dot(R) approximates the
           original K
    W    - is a matrix such that W.T.dot(K_chol.dot(W)) approximates the
           original K
    
    """
    assert(eta>0 and eta<1)
    assert(power>=0)
    assert(blocksize>=0)
    assert(len(X)>=0)
    
    m=len(X)

    # growing low rank basis
    R=zeros((blocksize,m))
    
    # diagonal (assumed to be one)
    d=ones(m)
    
    # used indices
    I=[]
    nu=[]
    
    # algorithm is executed as long as a is bigger than eta precision
    a=d.max()
    I.append(d.argmax())
    
    # growing set of evaluated kernel values
    K=zeros((blocksize,m))
    
    j=0
    while a>eta:
        nu.append(sqrt(a))
        
        if power>=1:
            K[j,:]=kernel.kernel([X[I[j]]], X)**power
        else:
            K[j,:]=ones(m)
            
        if j==0:
            R_dot_j=0
        elif j==1:
            R_dot_j=R[:j,:]*R[:j,I[j]]
        else:
            R_dot_j=R[:j,:].T.dot(R[:j,I[j]])
                        
        R[j,:]=(K[j,:] - R_dot_j)/nu[j]
        d=d-R[j,:]**2
        a=d.max()
        I.append(d.argmax())
        j=j+1
        
        # allocate more space for kernel
        if j>=len(K):
            K=vstack((K, zeros((blocksize,m))))
            R=vstack((R, zeros((blocksize,m))))
            
    # remove un-used rows which were located unnecessarily
    K=K[:j,:]
    R=R[:j,:]

    # remove list pivot index since it is not used
    I=I[:-1]
    
    # from low rank to full rank
    W=solve(R[:,I], R)
    
    # low rank K
    K_chol=K[:,I]
    
    return K_chol, I, R, W
