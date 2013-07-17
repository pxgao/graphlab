"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.lib.twodim_base import eye
from numpy.linalg.linalg import cholesky
from numpy.ma.core import reshape, shape
from sets import Set
from src.GraphlabLines import GraphlabLines
import os

class PrecomputeDenseMatrixKernelBP(object):
    """
    Class to pre-compute KernelBP matrices and systems that are stored as dense
    matrices and can be stored in memory.
    
    All parameters are of the form as return by the ToyModel class.
    
    graph - a graph dictionary which contains a list of neighbours for every node
    edges - a list of ordered pairs which represent the "depends on" structure
    data - a dictionary which maps an edge (pair) to a pair of data (for source
           and target node)
    kernel - kernel instance to use
    output_filename - filename of the output graph, all matrices will be stored
                      in the same directory
    
    """
    def __init__(self, graph, edges, data, observations, kernel, reg_lambda, output_filename):
        self.graph=graph
        self.edges=edges
        self.data=data
        self.observations=observations
        self.kernel=kernel
        self.reg_lambda=reg_lambda
        self.output_filename=output_filename
        
    def precompute(self):
        # collect lines for Graphlab graph definition file for full rank case
        graphlab_lines=GraphlabLines(output_filename=self.output_filename)
                                            
        # compute all non-symmetric kernels for incoming messages at a node
        print "precomputing (non-symmetric) kernels for incoming messages at a node"
        graphlab_lines.lines.append("# non-observed nodes")
        for node in self.graph:
            added_node=False
            
            for in_message in self.graph[node]:
                for out_message in self.graph[node]:
                    if in_message==out_message:
                        continue
                    
                    # dont add nodes which have no kernels, and only do once if they have
                    if not added_node:
                        graphlab_lines.new_non_observed_node(node)
                        added_node=True
                        
                    edge_in_message=(node, in_message)
                    edge_out_message=(out_message, node)
                    
                    lhs=self.data[edge_in_message][0]
                    rhs=self.data[edge_out_message][1]
                    lhs=reshape(lhs, (len(lhs),1))
                    rhs=reshape(rhs, (len(rhs),1))
                    K=self.kernel.kernel(lhs,rhs)
                    graphlab_lines.add_non_observed_node(node, out_message, in_message, K)
            
        print "precomputing kernel (vectors) at observed nodes"
        graphlab_lines.lines.append(os.linesep + "# observed nodes")
        for node, observation in self.observations.items():
            graphlab_lines.new_observed_node(node)
            
            for out_message in self.graph[node]:
                edge=(out_message, node)
                lhs=self.data[edge][1]
                lhs=reshape(lhs, (len(lhs), 1))
                rhs=[[observation]]
                K=self.kernel.kernel(lhs, rhs)
                graphlab_lines.add_observed_node(node, out_message, K)
                
        
        # now precompute systems for inference
        
        print "precomputing systems for messages from observed nodes"
        graphlab_lines.lines.append(os.linesep + "# edges with observed targets")
        for node, observation in self.observations.items():
            for out_message in self.graph[node]:
                edge=(out_message, node)
                graphlab_lines.new_edge_observed_target(node, out_message)
                
                data_source=self.data[edge][0]
                data_source=reshape(data_source, (len(data_source), 1))
                data_target=self.data[edge][1]
                data_target=reshape(data_target, (len(data_target), 1))
        
                Ks=self.kernel.kernel(data_source)
                Kt=self.kernel.kernel(data_target)
                
                Ls=cholesky(Ks+eye(shape(Ks)[0])*self.reg_lambda)
                Lt=cholesky(Kt+eye(shape(Kt)[0])*self.reg_lambda)
                
                graphlab_lines.add_edge(node, out_message,"L_s", Ls)
                graphlab_lines.add_edge(node, out_message,"L_t", Lt)
        
        print "precomputing systems for messages from non-observed nodes"
        graphlab_lines.lines.append(os.linesep + "# edges with non-observed targets")
        for edge in self.edges:
            # exclude edges which involve observed nodes
            is_edge_target_observed=len(Set(self.observations.keys()).intersection(Set(edge)))>0
            if not is_edge_target_observed:
                graphlab_lines.new_edge_observed_target(edge[1], edge[0])
                
                data_source=self.data[edge][0]
                data_source=reshape(data_source, (len(data_source), 1))
                Ks=self.kernel.kernel(data_source)
                Ls=cholesky(Ks+eye(shape(Ks)[0])*self.reg_lambda)
                graphlab_lines.add_edge(edge[1], edge[0],"L_s", Ls)
                
        # write graph definition file to disc
        graphlab_lines.flush()