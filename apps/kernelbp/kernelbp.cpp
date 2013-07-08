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
#include "global_parameters.hpp"

struct edge_betas_type 
{
	map<pair<vertex_id_type,vertex_id_type>, VectorXd> betas;
	
	edge_betas_type & operator+=(const edge_betas_type & other)
	{
		for(map<pair<vertex_id_type,vertex_id_type>, VectorXd>::const_iterator it=other.betas.begin(); it!=other.betas.end(); ++it)
			betas[it->first]=it->second;
		
		return *this;
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
	distributed_control dc;


	global_logger().set_log_level(LOG_WARNING);
	global_logger().set_log_to_console(true);
  
	const string description = "Kernel Belief Propagation";
	command_line_options clopts(description);
	string graph_filename;
	string output_filename;
	string exec_type = "sync";
	clopts.attach_option("graph_filename", graph_filename, "The filename of the the graph definition file.");
	clopts.add_positional("graph_filename");
	clopts.attach_option("output_filename", output_filename, "The directory in which to save the final messages.");
	clopts.add_positional("output_filename");
	clopts.attach_option("beta_epsilon", BETA_EPSILON, "The tolerance level for message norm convergence.");
	clopts.attach_option("engine", exec_type, "The type of engine to use {async, sync}.");
	
	if (!clopts.parse(argc, argv))
	{
		mpi_tools::finalize();
		return clopts.is_set("help")? EXIT_SUCCESS : EXIT_FAILURE;
	}

	if (graph_filename.empty())
	{
		logstream(LOG_ERROR) << "No graph definition file was provided." << endl;
		clopts.print_description();
		return EXIT_FAILURE;
	}
	else
	{
		// extract graph dir
		unsigned index=graph_filename.find_last_of("/");
		if (index>=0)
			GRAPH_DIR = graph_filename.substr(0, index+1);
		else
			GRAPH_DIR="";
	}
	
	if (output_filename.empty())
	{
		logstream(LOG_ERROR) << "No output filename was provided." << endl;
		clopts.print_description();
		return EXIT_FAILURE;
	}
	
	
	// start webserver
	launch_metric_server();
	
	// load graph and engine
	graph_type graph(dc, clopts);
	graph.load(graph_filename, line_parser);
	omni_engine<program> engine(dc, graph, "sync");
	
	// signal all nodes to start kernel BP
	engine.signal_all();
	
	// start kernelbp program
	engine.start();
	
	// map-reduce betas
 	edge_betas_type edge_betas=graph.map_reduce_edges<edge_betas_type>(collect_edge_betas);
 	
 	// save betas to output file
	ofstream outfile;
	outfile.open(output_filename.c_str());
	if (outfile.fail())
		cerr << "file \"" << output_filename << "\" could not be opened/created!" << endl;

 	for (map<pair<vertex_id_type,vertex_id_type>, VectorXd>::iterator it=edge_betas.betas.begin(); it!=edge_betas.betas.end(); ++it)
		outfile << edge_betas.betas;
		
	stop_metric_server();
	mpi_tools::finalize();
	return EXIT_SUCCESS;

	return 0;
}

