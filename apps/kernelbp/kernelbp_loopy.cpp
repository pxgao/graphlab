#include <graphlab.hpp>
using namespace graphlab;

#include <vector>
#include <set>
#include <string>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include "graph_type.hpp"
#include "line_parser.hpp"
#include "helpers.hpp"
#include "gather_type.hpp"
#include "program.hpp"

struct edge_betas_type 
{
	map<pair<vertex_id_type,vertex_id_type>, VectorXd> betas;
	
	edge_betas_type & operator+=(const edge_betas_type & other)
	{
		for(map<pair<vertex_id_type,vertex_id_type>, VectorXd>::const_iterator it=other.betas.begin(); it!=other.betas.end(); ++it)
			betas[it->first]=it->second;
	}
	
	void save(oarchive& oarc) const
	{
		oarc << betas;
	}
	
	void load(iarchive& iarc)
	{
		iarc >> betas;
	}
};

edge_betas_type collect_edge_betas(const graph_type::edge_type& edge)
{
	edge_betas_type eb;
	eb.betas[pair<vertex_id_type,vertex_id_type>(edge.source().id(), edge.target().id())]=edge.data().beta;
	return eb;
}
      
      
int main(int argc, char** argv)
{
	mpi_tools::init(argc, argv);
	
	// load graph and engine
	distributed_control dc;
	graph_type graph(dc);
	graph.load("graph_ascii_files_ichol.txt", line_parser);
	omni_engine<program> engine(dc, graph, "sync");
	
	// signal all nodes to start kernel BP
	//engine.signal_vset(graph.select(is_vertex_1));
	engine.signal_vset(graph.complete_set());
	
	// start program
	engine.start();

	// let all the parallel stuff finish
	mpi_tools::finalize();
	
	// map-reduce betas
 	edge_betas_type edge_betas=graph.map_reduce_edges<edge_betas_type>(collect_edge_betas);
 	cout << edge_betas.betas << endl;

	return 0;
}

