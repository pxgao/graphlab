#ifndef GRAPH_TYPE_H
#define GRAPH_TYPE_H

#include <graphlab.hpp>
using namespace graphlab;

#include <vector>
#include <string>
#include <utility>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include "stream_operators.hpp"
#include "ascii_matrix_loader.hpp"

struct node_data
{
	// the kernel of data with observation all incoming edges (first key, outgoing message) and outgoing edges (second key, incoming messages)
    map<vertex_id_type, VectorXd> kernel_dict_obs;
    map<pair<vertex_id_type,vertex_id_type>, MatrixXd> kernel_dict;
    
    // map of multiplied incoming messages for each target node (used for apply-scatter)
    map<vertex_id_type, VectorXd> multiplied_incoming_messages;
    
    bool is_observed;
    
	node_data()
	{
		is_observed=false;
	}
	
	void save(oarchive& oarc) const
	{
		oarc << kernel_dict_obs;
		oarc << kernel_dict;
		oarc << multiplied_incoming_messages;
		oarc << is_observed;
	}
	
	void load(iarchive& iarc)
	{
		iarc >> kernel_dict_obs;
		iarc >> kernel_dict;
		iarc >> multiplied_incoming_messages;
		iarc >> is_observed;
	}
};

struct edge_data
{
	// kernel matrix of joint data
	map<string, MatrixXd> solution_matrices;
	
	// beta message that corresponds to vertices 
	VectorXd beta;
	
	// full rank or incomplete cholesky
	bool full_rank;
	
	edge_data()
	{
		full_rank=true;
	}
	
	void save(oarchive& oarc) const
	{
		oarc << solution_matrices;
		oarc << beta;
		oarc << full_rank;
	}
	
	void load(iarchive& iarc)
	{
		iarc >> solution_matrices;
		iarc >> beta;
		iarc >> full_rank;
	}
};

typedef distributed_graph<node_data, edge_data> graph_type;

ostream& operator <<(ostream& out, const node_data& n) 
{
	out << "node_data: ";
	out << "kernel_dict_obs=\n" << n.kernel_dict_obs << endl;
	out << "kernel_dict=\n" << n.kernel_dict << endl;
	out << "is_observed=\n" << n.is_observed << endl;
	return out;
}

ostream& operator <<(ostream& out, const edge_data& e) 
{
	out << "edge_data: ";
	out << "solution_matrices=\n" << e.solution_matrices << endl;
	out << "beta=\n" << e.beta << endl;
	out << endl;
	return out;
}

#endif //GRAPH_TYPE_H
