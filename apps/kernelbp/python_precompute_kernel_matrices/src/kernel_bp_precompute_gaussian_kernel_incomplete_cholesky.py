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

The Graphlab implementation offers the possibility of using incomplete Cholesky
factorisation of involved kernels and features in order to reach constant time
message updates. This script illustrates how to change pre-computed kernels and
linear system solutions to do that, see documentation for details.
"""

from numpy.lib.npyio import savetxt
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from scipy.linalg import qr
from numpy.ma.core import shape
from numpy.random import randint
from sets import Set
from src.GaussianKernel import GaussianKernel
from src.GraphlabLines import GraphlabLines
from src.ToyModel import ToyModel
from src.incomplete_cholesky import incomplete_cholesky
import os.path

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
n_min=10
n_max=100
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

# collect lines for Graphlab graph definition file for full rank case
graphlab_lines=GraphlabLines(output_filename="graph_definition.txt", \
                             caller_filename=os.path.basename(__file__))


# regulariser for matrix inversions
reg_lambda=0.1

# incomplete cholesky cutoff parameter
eta=0.1

# store index sets at source and target of every edge
index_sets={}

print "precomputing systems for messages from observed nodes"
graphlab_lines.lines.append(os.linesep + "# edges with observed targets")
for node, observation in observations.items():
    for out_message in graph[node]:
        edge=(out_message, node)
        graphlab_lines.new_edge_observed_target(node, out_message)
        
        data_source=data[edge][0]
        data_target=data[edge][1]
        Ks_chol, Is, Rs, Ws=incomplete_cholesky(data_source, kernel, eta)
        Kt_chol, It, Rt, Wt=incomplete_cholesky(data_target, kernel, eta)
        
        Qs,Rs,Ps=qr(Ws.dot(Ws.T)+Ks_chol+eye(shape(Ks_chol)[0])*reg_lambda, pivoting=True)
        Qt,Rt,Pt=qr(Wt.dot(Wt.T)+Kt_chol+eye(shape(Kt_chol)[0])*reg_lambda, pivoting=True)
        
        savetxt(graphlab_lines.add_edge(node, out_message,"Q_s"), Qs)
        savetxt(graphlab_lines.add_edge(node, out_message,"R_s"), Rs)
        savetxt(graphlab_lines.add_edge(node, out_message,"P_s"), Ps)
        
        savetxt(graphlab_lines.add_edge(node, out_message,"Q_t"), Qt)
        savetxt(graphlab_lines.add_edge(node, out_message,"R_t"), Rt)
        savetxt(graphlab_lines.add_edge(node, out_message,"P_t"), Pt)
        
        savetxt(graphlab_lines.add_edge(node, out_message,"W"), Ws.dot(Wt.T))

print "precomputing systems for messages from non-observed nodes"
graphlab_lines.lines.append(os.linesep + "# edges with non-observed targets")
for edge in edges:
    # exclude edges which involve observed nodes
    is_edge_target_observed=len(Set(observations.keys()).intersection(Set(edge)))>0
    if not is_edge_target_observed:
        graphlab_lines.new_edge_observed_target(edge[1], edge[0])
        
        data_source=data[edge][0]
        Ks=kernel.kernel(data_source)
        Ls=cholesky(Ks+eye(shape(Ks)[0])*reg_lambda)
        
        Ls_filename=graphlab_lines.add_edge(edge[1], edge[0],"L_s")
#        print Ls_filename
        savetxt(Ls_filename, Ls)

print "precomputing (non-symmetric) kernels for incoming messages at a node"
graphlab_lines.lines.append("# non-observed nodes")
for node in graph:
    added_node=False
    
    for in_message in graph[node]:
        for out_message in graph[node]:
            if in_message==out_message:
                continue
            
            # dont add nodes which have no kernels, and only do once if they have
            if not added_node:
                graphlab_lines.new_non_observed_node(node)
                added_node=True
                
            edge_in_message=(node, in_message)
            
            edge_out_message=(out_message, node)
            
            lhs=data[edge_in_message][0]
            rhs=data[edge_out_message][1]
            K=kernel.kernel(lhs,rhs)

            K_filename=graphlab_lines.add_non_observed_node(node, out_message, in_message)
            
#            print K_filename
            savetxt(K_filename, K)
    
print "precomputing kernel (vectors) at observed nodes"
graphlab_lines.lines.append(os.linesep + "# observed nodes")
for node, observation in observations.items():
    graphlab_lines.new_observed_node(node)
    
    for out_message in graph[node]:
        edge=(out_message, node)
        K=kernel.kernel(data[edge][1], [observation])

        K_filename=graphlab_lines.add_observed_node(node, out_message)
#        print K_filename
        savetxt(K_filename, K)

        
# write graph definition file to disc
graphlab_lines.flush()