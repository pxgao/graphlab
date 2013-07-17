"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""


from numpy.lib.npyio import savetxt
import datetime
import os

class GraphlabLines(object):
    def __init__(self, output_filename, filename_suffix=""):
        self.filename_suffix=filename_suffix
        self.output_filename=output_filename
        self.lines=[]
        self.lines.append("# Auto-generated graph definition file for Graphlab Kernel-BP implementation.")
        self.lines.append("# Generated at " + \
                datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        self.lines.append("")
        
        # extract foldername
        if self.output_filename.find(os.sep) is not -1:
            self.pathname=os.sep.join(self.output_filename.split(os.sep)[:-1])
        else:
            self.pathname=""
            
            
        # create folder if not yet existing
        try:
            os.makedirs(self.pathname)
        except OSError:
            pass
        
    def flush(self):
        f=open(self.output_filename, "w")
        f.write(os.linesep.join(self.lines))
        f.close()
        
    def new_non_observed_node(self, node):
        self.lines.append("non_observed_node " + str(node) + "\t\t")
   
    def add_non_observed_node(self, node, out_message, in_message, K):
        filename=str(in_message) + "->" + str(node)+ "->" + str(out_message) + \
                "_non_obs_kernel" + self.filename_suffix + ".txt"
        self.lines[-1]+=str(in_message) + " " + str(out_message) + " " + filename + "\t"
        
        if self.pathname is not "":
            filename=self.pathname + os.sep + filename
        savetxt(filename, K)
    
    def new_observed_node(self, node):
        self.lines.append("observed_node " + str(node) + "\t\t")
    
    def add_observed_node(self, node, out_message, K):
        filename=str(out_message)+ "->" + str(node) + "_obs_kernel" + self.filename_suffix + ".txt"
        self.lines[-1]+=str(out_message) + " " + filename + "\t"
        
        if self.pathname is not "":
            filename=self.pathname + os.sep + filename
        savetxt(filename, K)
    
    def new_edge_observed_target(self, node, out_message):
        self.lines.append("edge_observed_target " + str(out_message) + \
                          " " + str(node) + "\t\t")

    def new_edge_non_obserbed_target(self, node, out_message):
        self.lines.append("edge_non_observed_target " + str(out_message) + \
                          " " + str(node) + "\t\t")
    
    def add_edge(self, node, out_message, matrix_name, K):
        filename=str(out_message)+ "->" + str(node) + "_" + matrix_name + self.filename_suffix + ".txt"
        self.lines[-1]+=matrix_name + " " + filename + " "
        
        if self.pathname is not "":
            filename=self.pathname + os.sep + filename
        savetxt(filename, K)
