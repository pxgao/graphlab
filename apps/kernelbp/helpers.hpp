#ifndef HELPERS_H
#define HELPERS_H

#include <list>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

bool print_summary(const graph_type::vertex_type& vertex)
{
	cout << "print_summary(), vid=" << vertex.id();
	cout << endl;
	return 0;
}

bool is_observed(const graph_type::vertex_type& vertex)
{
	return !vertex.num_out_edges();
}

bool is_vertex_1(const graph_type::vertex_type& vertex)
{
	return vertex.id()==1;
}

bool is_root(const graph_type::vertex_type& vertex)
{
	bool result=vertex.num_out_edges() && !vertex.num_in_edges();
	//cout << "vid=" << vertex.id() << " is ";
	//if (result)
	//	cout << "a leaf";
	//else
	//	cout << "no leaf";
	
	//cout << endl;
	return result;
}




#endif //HELPERS_H
