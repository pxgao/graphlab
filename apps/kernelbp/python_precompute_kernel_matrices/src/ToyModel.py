"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from copy import deepcopy
from numpy.ma.core import ones
from numpy.random import randn, rand
from sets import Set

class ToyModel(object):
    """
    Defines a graphical model
    
    p(x1)p(x2|x1)p(x3|x1)p(x4|x2,x3)p(x5|x3)
    
    where the individual distributions are defined via various sample methods
    """
        
    @staticmethod
    def sample_real(n=1, mean_top=[0, 2], std_top=0.2, weights_top=[0.5, 0.5], \
               std_middle=0.2, std_bottom=0.2, weights_bottom=[0.5, 0.5]):
        """
        This code samples n times from the model

        p(x1)p(x2|x1)p(x3|x1)p(x4|x2,x3)p(x5|x3), where all distributions' 
        domains are the real line, i.e.,
        
        p(x1)                 - mixture of 2 Gaussians with fixed means and
                                fixed variance
        p(x2|x1),p(x3|x1)     - Gaussian whose mean is given by sample from x1,
                                fixed variance
        p(x4|x2,x3)           - mixture of 2 Gaussians whose mean is either
                                0.5*(x2+x3) or zero, fixed variance
        p(x5|x3)              - mixture of 2 Gaussians whose mean is either x3
                                or zero, fixed variance
                                
        returns a dictionary {node_ind -> samples}
        """
        assert(sum(weights_top) == 1)
        assert(sum(weights_bottom) == 1)
        
        
        mean_x1 = ones(n) * mean_top[0]
        mean_x1[rand(n) < weights_top[0]] = mean_top[1]
        x1 = mean_x1 + randn(n) * std_top
        
        x2 = x1 + randn(n) * std_middle
        x3 = x1 + randn(n) * std_middle
        
        mean_x4 = (x2 + x3) * 0.5
        mean_x4[rand(n) < weights_bottom[0]] = 0
        x4 = mean_x4 + randn(n) * std_bottom
        
        mean_x5 = deepcopy(x3)
        mean_x5[rand(n) < weights_bottom[0]] = 0
        x5 = mean_x5 + randn(n) * std_bottom
        
        return {1:x1, 2:x2, 3:x3, 4:x4, 5:x5}
    
    @staticmethod
    def get_moralised_graph():
        # construct nodes and their neighbours
        graph = {}
        graph[1] = [2, 3]
        graph[2] = [1, 3, 4]
        graph[3] = [1, 2, 4, 5]
        graph[4] = [2, 3]
        graph[5] = [3]

        return graph
    
    @staticmethod
    def extract_edges(observations):
        """
        Returns edges of the moralised undirected graph that is used by the
        Graphlab Kernel-BP implementation. It is represented via directed edges
        which implement a "depends on" semantic. Observed nodes do not have
        outgoing edges (the rest of the undirected edges is represented via two
        directed ones)
        
        Parameters:
        observations - is a dictionary of observed nodes and their observation
        """
        edges = Set()
        graph = ToyModel.get_moralised_graph()
        for node in graph.keys():
            for neighbour in graph[node]:
                # observations nodes do not have outgoing edges
                if node not in observations:
                    edges.add((node, neighbour))
                    
                if neighbour not in observations:
                    edges.add((neighbour, node))

        return list(edges)