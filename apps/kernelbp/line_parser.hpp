#ifndef LINE_PARSER_H
#define LINE_PARSER_H

#include <graphlab.hpp>
using namespace graphlab;

#include <vector>
#include <string>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

//#include "stream_operators.hpp"
#include "ascii_matrix_loader.hpp"
#include "graph_type.hpp"
#include "global_parameters.hpp"

bool line_parser(graph_type& graph, const string& filename, const string& textline)
{
	//cout << "filename=" << filename << endl;
	//cout << "textline=" << textline << endl;
	
	stringstream strm(textline);

	string type;
	strm >> type;
	cout << "type=" << type << endl;
	
	if (!type.compare("non_observed_node") || !type.compare("observed_node"))
	{
		node_data n;
		
		vertex_id_type vid;
		strm >> vid;
		cout << "vid=" << vid << endl;
		
		if (!type.compare("non_observed_node"))
		{
			n.is_observed=false;
		
			// read kernel matrices
			while(true)
			{
				vertex_id_type vid_source;
				strm >> vid_source;
				cout << "vid_source=" << vid_source << endl;
				
				if (strm.fail())
					break;
				
				vertex_id_type vid_target;
				strm >> vid_target;
				cout << "vid_target=" << vid_target << endl;
		
				string filename;
				strm >> filename;
				cout << "filename=" << filename << endl;
			
				MatrixXd K=load_ascii_matrix(filename, GRAPH_DIR);
				
				// add kernel to inner map
				n.kernel_dict[pair<vertex_id_type,vertex_id_type>(vid_source,vid_target)]=K;
				cout << "K_" << vid << "^(" << vid_source << "," << vid_target << "):" << endl << K << endl;
			}
		}
		else if (!type.compare("observed_node"))
		{
			n.is_observed=true;
		
			// read kernel matrices
			while (true)
			{
				vertex_id_type vid_source;
				strm >> vid_source;
				cout << "vid_source=" << vid_source << endl;
				
				if (strm.fail())
					break;
				
				string filename;
				strm >> filename;
				cout << "filename=" << filename << endl;
			
				VectorXd kernel=load_ascii_matrix(filename, GRAPH_DIR);
				n.kernel_dict_obs[vid_source]=kernel;
				cout << "k_" << vid << ": " << kernel << endl;
			}
		}

		// add node_data
		graph.add_vertex(vid, n);
	}
	else if (!type.compare("edge_non_observed_target") || !type.compare("edge_observed_target"))
	{
		edge_data e;
		
		// read edge directions
		vertex_id_type source;
		vertex_id_type target;
		strm >> source;
		strm >> target;
		
		// read all precomputed systems' matrices
		while (true)
		{
			string name, filename;
			strm >> name;
			strm >> filename;
			
			if (strm.fail())
				break;
				
			MatrixXd M=load_ascii_matrix(filename, GRAPH_DIR);
			e.solution_matrices[name]=M;
			cout << name << ": " << filename << ": " << M << endl;
		}
		
		// in case of full rank there is a cholesky factor
		if (e.solution_matrices.find("L_s")==e.solution_matrices.end())
			e.full_rank=false;
		else
			e.full_rank=true;
			
		cout << "full rank: " << e.full_rank << endl;
		
		// add edge data
		graph.add_edge(source, target, e);
	}
	else if (!type.substr(0,1).compare("#") || !type.substr(0,1).compare("/") || !type.substr(0,1).compare("%"))
	{
		// comment fields
	}
	else
	{
		cout << "type=" << type << " is unknown!" << endl;
	}
	
	// cout << endl;
	return true;
}

#endif //LINE_PARSER_H
