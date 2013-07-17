"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.lib.npyio import loadtxt
from numpy.linalg.linalg import norm
from numpy.ma.core import asarray, sqrt, shape, reshape
from src.GaussianKernel import GaussianKernel
from src.PrecomputeDenseMatrixKernelBP import PrecomputeDenseMatrixKernelBP
from src.ToyModel import ToyModel
import os
import subprocess
import unittest

class PrecomputeDenseMatrixKernelTests(unittest.TestCase):
    output_filename=".." + os.sep + "graph_test1" + os.sep + "graph.txt"
    output_folder=os.sep.join(output_filename.split(os.sep)[:-1])+os.sep
    
    def test_all(self):
        self.fixed_data()
        self.graphlab_results()
    
    def fixed_data(self):
        """
        Uses some fixed data from the ToyModel and pre-computes all matrices.
        Results are asserted to be correct (against Matlab implementation) and
        output files can be used to test the KernelBP implementation.
        """
        model=ToyModel()
        graph=model.get_moralised_graph()
        
        # one observation at node 4
        observations={4:0.0}
        
        # directed edges for kernel BP implementation
        edges=model.extract_edges(observations)
        
        print "graph:", graph
        print "observations:", observations
        print "edges:", edges
        
        # we sample the data jointly, so edges will share data along vertices
        joint_data={}
        joint_data[1]=[ -0.274722354853981, 0.044011207316815,  0.073737451640458]
        joint_data[2]=[ -0.173264814517908,  0.213918664844409,  0.123246012188621]
        joint_data[3]=[ -0.348879413536605,  -0.081766464397055,   -0.117171083361484]
        joint_data[4]=[  -0.014012058355118, -0.145789276405117,   -0.317649695308685]
        joint_data[5]=[ -0.291794859908481,   0.260902212951398,  -0.276258182225143]
        
        # generate data in format that works for the dense matrix class, i.e., a pair of
        # points for every edge
        data={}
        for edge in edges:
            # only sample once per undirected edge
            inverse_edge=(edge[1], edge[0])
            data[edge]=(joint_data[edge[0]],joint_data[edge[1]])
            data[inverse_edge]=(joint_data[edge[1]],joint_data[edge[0]])
        
        # Gaussian kernel used in matlab files
        kernel=GaussianKernel(sigma=sqrt(0.15))
        
        # use the example class for dense matrix data that can be stored in memory

        precomputer=PrecomputeDenseMatrixKernelBP(graph, edges, data, observations, \
                                                  kernel, reg_lambda=0.1, \
                                                  output_filename=self.output_filename)
        
        precomputer.precompute()
        
        # go through all the files and make sure they contain the correct matrices
        
        # files created by matlab implementation
        filenames=[
                   "1->2->3_non_obs_kernel.txt",
                   "1->2->4_non_obs_kernel.txt",
                   "1->3->2_non_obs_kernel.txt",
                   "1->3->4_non_obs_kernel.txt",
                   "1->3->5_non_obs_kernel.txt",
                   "2->1->3_non_obs_kernel.txt",
                   "2->3->1_non_obs_kernel.txt",
                   "2->3->4_non_obs_kernel.txt",
                   "2->3->5_non_obs_kernel.txt",
                   "2->4->3_non_obs_kernel.txt",
                   "3->1->2_non_obs_kernel.txt",
                   "3->2->1_non_obs_kernel.txt",
                   "3->2->4_non_obs_kernel.txt",
                   "3->4->2_non_obs_kernel.txt",
                   "4->2->1_non_obs_kernel.txt",
                   "4->2->3_non_obs_kernel.txt",
                   "4->3->1_non_obs_kernel.txt",
                   "4->3->2_non_obs_kernel.txt",
                   "4->3->5_non_obs_kernel.txt",
                   "5->3->1_non_obs_kernel.txt",
                   "5->3->2_non_obs_kernel.txt",
                   "5->3->4_non_obs_kernel.txt",
                   "3->4_obs_kernel.txt",
                   "2->4_obs_kernel.txt",
                   "1->2_L_s.txt",
                   "1->3_L_s.txt",
                   "2->1_L_s.txt",
                   "2->3_L_s.txt",
                   "2->4_L_s.txt",
                   "2->4_L_t.txt",
                   "3->1_L_s.txt",
                   "3->2_L_s.txt",
                   "3->4_L_s.txt",
                   "3->4_L_t.txt",
                   "3->5_L_s.txt",
                   "5->3_L_s.txt"
                   ]
        
        # from matlab implementation
        matrices={
                    "2->1->3_non_obs_kernel.txt": asarray([[1.000000, 0.712741, 0.667145], [0.712741, 1.000000, 0.997059], [0.667145, 0.997059, 1.000000]]),
                    "3->1->2_non_obs_kernel.txt": asarray([[1.000000, 0.712741, 0.667145], [0.712741, 1.000000, 0.997059], [0.667145, 0.997059, 1.000000]]),
                    "1->2->3_non_obs_kernel.txt": asarray([[1.000000, 0.606711, 0.745976], [0.606711, 1.000000, 0.972967], [0.745976, 0.972967, 1.000000]]),
                    "1->2->4_non_obs_kernel.txt": asarray([[1.000000, 0.606711, 0.745976], [0.606711, 1.000000, 0.972967], [0.745976, 0.972967, 1.000000]]),
                    "3->2->1_non_obs_kernel.txt": asarray([[1.000000, 0.606711, 0.745976], [0.606711, 1.000000, 0.972967], [0.745976, 0.972967, 1.000000]]),
                    "3->2->4_non_obs_kernel.txt": asarray([[1.000000, 0.606711, 0.745976], [0.606711, 1.000000, 0.972967], [0.745976, 0.972967, 1.000000]]),
                    "4->2->1_non_obs_kernel.txt": asarray([[1.000000, 0.606711, 0.745976], [0.606711, 1.000000, 0.972967], [0.745976, 0.972967, 1.000000]]),
                    "4->2->3_non_obs_kernel.txt": asarray([[1.000000, 0.606711, 0.745976], [0.606711, 1.000000, 0.972967], [0.745976, 0.972967, 1.000000]]),
                    "1->3->2_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "1->3->4_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "1->3->5_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "2->3->1_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "2->3->4_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "2->3->5_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "4->3->1_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "4->3->2_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "4->3->5_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "5->3->1_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "5->3->2_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "5->3->4_non_obs_kernel.txt": asarray([[1.000000, 0.788336, 0.836137], [0.788336, 1.000000, 0.995830], [0.836137, 0.995830, 1.000000]]),
                    "2->4->3_non_obs_kernel.txt": asarray([[1.000000, 0.943759, 0.735416], [0.943759, 1.000000, 0.906238], [0.735416, 0.906238, 1.000000]]),
                    "3->4->2_non_obs_kernel.txt": asarray([[1.000000, 0.943759, 0.735416], [0.943759, 1.000000, 0.906238], [0.735416, 0.906238, 1.000000]]),
                    "2->4_obs_kernel.txt": asarray([[0.999346], [0.931603], [0.714382]]),
                    "2->4_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.578476, 0.874852, 0.000000], [0.711260, 0.641846, 0.426782]]),
                    "2->4_L_t.txt": asarray([[1.048809, 0.000000, 0.000000], [0.899839, 0.538785, 0.000000], [0.701191, 0.510924, 0.589311]]),
                    "3->4_obs_kernel.txt": asarray([[0.999346], [0.931603], [0.714382]]),
                    "3->4_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.751649, 0.731453, 0.000000], [0.797226, 0.542204, 0.412852]]),
                    "3->4_L_t.txt": asarray([[1.048809, 0.000000, 0.000000], [0.899839, 0.538785, 0.000000], [0.701191, 0.510924, 0.589311]]),
                    "2->1_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.578476, 0.874852, 0.000000], [0.711260, 0.641846, 0.426782]]),
                    "3->1_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.751649, 0.731453, 0.000000], [0.797226, 0.542204, 0.412852]]),
                    "1->2_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.679572, 0.798863, 0.000000], [0.636098, 0.706985, 0.442211]]),
                    "3->2_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.751649, 0.731453, 0.000000], [0.797226, 0.542204, 0.412852]]),
                    "1->3_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.679572, 0.798863, 0.000000], [0.636098, 0.706985, 0.442211]]),
                    "2->3_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.578476, 0.874852, 0.000000], [0.711260, 0.641846, 0.426782]]),
                    "5->3_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.344417, 0.990645, 0.000000], [0.952696, 0.054589, 0.435191]]),
                    "3->5_L_s.txt": asarray([[1.048809, 0.000000, 0.000000], [0.751649, 0.731453, 0.000000], [0.797226, 0.542204, 0.412852]])
                    }
        
        assert(len(filenames)==len(matrices))
        
        for filename in filenames:
            self.assertTrue(self.assert_file_matrix(self.output_folder+filename, matrices[filename]))
        
    def graphlab_results(self):
        """
        Based on the fixed data from above, calls graphlab and asserts that it
        outputs some fixed beta coefficients after convergence.
        This is a nasty unit test which compiles graphlab and runs it locally.
        """
        
        # run graphlab as system call from folder within
        folder=".."+os.sep+".."+os.sep+".."+os.sep+".."+os.sep+"debug"+os.sep+"apps"+os.sep+"kernelbp" + os.sep
        
        # cd to kernelbp binary directory
        os.chdir(folder)
        
        # compile kernelbp in case it hasnt been done yet
        self.assertEqual(subprocess.call(["make"]), 0)
        
        # run command:
        # mpiexec -n 1 kernelbp --graph_filename ../../../apps/kernelbp/python_precompute_kernel_matrices/unitttests/graph_test1/graph.txt --output_filename outfile.txt --engine sync
        command="mpiexec"
        args=[
              "-n", "1",
              "kernelbp",
              "--graph_filename", "../../../apps/kernelbp/python_precompute_kernel_matrices/graph_test1/graph.txt"
              ]        
        self.assertEqual(subprocess.call([command] + args), 0)
        
        # open result file
        try:
            f=open("outfile.txt")
            lines=[line.strip() for line in f.readlines()]
            f.close()
        except IOError:
            self.assertTrue(False, "Graphlab output file could not be opened.")
        
        # read betas
        betas={}
        idx=0
        while idx<len(lines):
            # cut off () of the edge pair of graphlab output
            vertices=lines[idx].strip(")").strip("(").split(",")
            
            # read all lines until next vector or end
            idx+=1
            temp=[]
            while idx<len(lines) and lines[idx][0]!="(":
                temp.append(float(lines[idx]))
                idx+=1
            
            betas[(int(vertices[0]), int(vertices[1]))]=asarray(temp)
        
        # matlab results
        reference_betas={}
        reference_betas[(2, 1)]=asarray([  0.667559,   0.298557,  -0.682077])
        reference_betas[(3, 1)]=asarray([  0.789823,   0.045293,  -0.611660])
        reference_betas[(1, 2)]=asarray([  0.770158,  -0.621991,   0.141367])
        reference_betas[(3, 2)]=asarray([  0.885537,  -0.310062,  -0.345956])
        reference_betas[(1, 3)]=asarray([  0.808071,  -0.583016,   0.084344])
        reference_betas[(2, 3)]=asarray([  0.821792,   0.072258,  -0.565188])
        reference_betas[(5, 3)]=asarray([  0.743932,  -0.011267,  -0.668161])
        reference_betas[(2, 4)]=asarray([  0.388615,   0.532370,  -0.752037])
        reference_betas[(3, 4)]=asarray([  0.414404,   0.483776,  -0.770863])
        reference_betas[(3, 5)]=asarray([  0.746400,   0.609564,   0.267055])
        
        # assert that these are cloe to graphlab
        for edge,beta in reference_betas.items():
            difference=norm(beta-betas[edge])
            self.assertLessEqual(difference, 1.5e-2)
        
    def assert_file_matrix(self, filename, M):
        try:
            with open(filename):
                m=loadtxt(filename)
                
                # python loads vectors as 1d-arrays, but we want 2d-col-vectors
                if len(shape(m))==1:
                    m=reshape(m, (len(m), 1))
                    
                self.assertEqual(M.shape, m.shape)
                self.assertLessEqual(norm(m-M), 1e-5)
                return True
        except IOError:
            return False
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
