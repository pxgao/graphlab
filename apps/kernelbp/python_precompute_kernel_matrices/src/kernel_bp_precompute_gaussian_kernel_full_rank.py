"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann

This is an example of how to pre-compute kernel matrices and linear system
solutions for our Graphlab implementation of Kernel-BP, full rank case.
All used data are on the real line so we use the standard Gaussian kernel.

This code uses a very simple graphical model to generate data,

p(x1)p(x2|x1)p(x3|x1)p(x4|x2,x3)p(x5|x3), where

p(x1)                 - mixture of 2 Gaussians with fixed mean and variance
p(x2|x1),p(x3|x1)     - Gaussian whose mean is given by sample from x1, fixed
                        variance
p(x4|x2,x3)           - mixture of 2 Gaussians whose mean is either (x2,x3)'s
                        samples' mean or zero, fixed variance
p(x5|x3)              - mixture of 2 Gaussians whose mean is either x3's sample's
                        mean or zero, fixed variance

Since Kernel-BP uses separate samples for each connected pair of nodes, we sample
the above graphical model a different number of times for each edge.

The moralised (undirected) graph for (Kernel-BP) belief propagation is given as
     X1
    /  \
   X2---X3
    \  / \
     X4  X5
     
All edges are stored as two directed edges in the opposite direction. The
semantic of a directed edge (s,t) between two nodes is "s depends on t". This
means that Kernel-BP messages always flow from t to s.
NOTE: Outgoing edges from observed nodes are removed. 

We will assume one observation at node x4. The program generates samples and
then illustrates how to compute the needed kernel matrices and pre-computed
solutions to the linear systems to be solved by Kernel-BP. All matrices are
stored in separate files. In addition, a graph definition file is created, which
can be loaded by the provided line reader of our Graphlab Kernel-BP
implementation. See documentation for details.
"""

from numpy.random import randint
from src.GaussianKernel import GaussianKernel
from src.PrecomputeDenseMatrixKernelBP import PrecomputeDenseMatrixKernelBP
from src.ToyModel import ToyModel

model=ToyModel()
graph=model.get_moralised_graph()

# one observation at node 4
observations={4:0.0}

# directed edges for kernel BP implementation
edges=model.extract_edges(observations)

print "graph:", graph
print "observations:", observations
print "edges:", edges

# sample data, random number of samples for each edge
n_min=5
n_max=6
data={} 
for edge in edges:
    samples = model.sample_real(randint(n_min, n_max))
    
    # only sample once per undirected edge
    inverse_edge=(edge[1], edge[0])
    if (edge not in data and inverse_edge not in data):
        data1=samples[edge[0]]
        data2=samples[edge[1]]
        
        data[edge]=(data1,data2)
        data[inverse_edge]=(data2,data1)
        
# compute all (here Gaussian) kernels of node data at edges with themselves
kernel=GaussianKernel(sigma=1)

# use the example class for dense matrix data that can be stored in memory
precomputer=PrecomputeDenseMatrixKernelBP(graph, edges, data, observations, \
                                          kernel, reg_lambda=0.1, output_filename="graph/graph.txt")

precomputer.precompute()