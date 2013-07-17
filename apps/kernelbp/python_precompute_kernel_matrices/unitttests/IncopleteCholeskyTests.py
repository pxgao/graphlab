"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.linalg.linalg import norm
from numpy.ma.core import reshape, arange, shape
from numpy.random import randn
from src.GaussianKernel import GaussianKernel
from src.incomplete_cholesky import incomplete_cholesky
import unittest

class IncopleteCholeskyTests(unittest.TestCase):
    def test_1(self):
        kernel=GaussianKernel(sigma=10)
        X=reshape(arange(9.0), (3,3))
        K_chol, I, R, W=incomplete_cholesky(X, kernel, eta=0.8, power=2)
        K=kernel.kernel(X)
        
        self.assertEqual(len(I), 2)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 2)
        
        self.assertEqual(shape(K_chol), (len(I), len(I)))
        for i in range(len(I)):
            self.assertEqual(K_chol[i,i], K[I[i], I[i]])
            
        self.assertEqual(shape(R), (len(I), len(X)))
        self.assertAlmostEqual(R[0,0], 1.000000000000000)
        self.assertAlmostEqual(R[0,1], 0.763379494336853)
        self.assertAlmostEqual(R[0,2], 0.339595525644939)
        self.assertAlmostEqual(R[1,0], 0)
        self.assertAlmostEqual(R[1,1], 0.535992421608228)
        self.assertAlmostEqual(R[1,2], 0.940571570355992)
        
        self.assertEqual(shape(W), (len(I), len(X)))
        self.assertAlmostEqual(W[0,0], 1.000000000000000)
        self.assertAlmostEqual(W[0,1], 0.569858199525808)
        self.assertAlmostEqual(W[0,2], 0)
        self.assertAlmostEqual(W[1,0], 0)
        self.assertAlmostEqual(W[1,1], 0.569858199525808)
        self.assertAlmostEqual(W[1,2], 1)
    
    def test_2(self):
        kernel=GaussianKernel(sigma=2)
        X=reshape(arange(9.0), (3,3))
        K_chol, I, R, W=incomplete_cholesky(X, kernel, eta=0.999)
        K=kernel.kernel(X)
        
        self.assertEqual(len(I), 2)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 2)
        
        self.assertEqual(shape(K_chol), (len(I), len(I)))
        for i in range(len(I)):
            self.assertEqual(K_chol[i,i], K[I[i], I[i]])
            
        self.assertEqual(shape(R), (len(I), len(X)))
        self.assertAlmostEqual(R[0,0], 1.000000000000000)
        self.assertAlmostEqual(R[0,1],  0.034218118311666)
        self.assertAlmostEqual(R[0,2], 0.000001370959086)
        self.assertAlmostEqual(R[1,0], 0)
        self.assertAlmostEqual(R[1,1], 0.034218071400058)
        self.assertAlmostEqual(R[1,2], 0.999999999999060)
        
        self.assertEqual(shape(W), (len(I), len(X)))
        self.assertAlmostEqual(W[0,0], 1.000000000000000)
        self.assertAlmostEqual(W[0,1], 0.034218071400090)
        self.assertAlmostEqual(W[0,2], 0)
        self.assertAlmostEqual(W[1,0], 0)
        self.assertAlmostEqual(W[1,1], 0.034218071400090)
        self.assertAlmostEqual(W[1,2], 1)
        
    def test_3(self):
        kernel=GaussianKernel(sigma=10)
        X=randn(3000,10)
        K_chol, I, R, W=incomplete_cholesky(X, kernel, eta=0.001)
        K=kernel.kernel(X)
        
        self.assertEqual(shape(K_chol), (len(I), (len(I))))
        self.assertEqual(shape(R), (len(I), (len(X))))
        self.assertEqual(shape(W), (len(I), (len(X))))
        
        self.assertLessEqual(norm(K-R.T.dot(R)), 1)
        self.assertLessEqual(norm(K-W.T.dot(K_chol.dot(W))), 1)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()