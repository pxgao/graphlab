#ifndef KERNELBP_PROGRAM
#define KERNELBP_PROGRAM

#include <limits>
#include "global_parameters.hpp"

class program : public ivertex_program<graph_type, gather_type>, public IS_POD_TYPE
{
protected:
	// apply stores incoming messages in here (key is where the message is from)
	map<vertex_id_type,VectorXd> message_map;

public:
	edge_dir_type gather_edges(icontext_type& context, const vertex_type& vertex) const
	{
		// incoming messages are from nodes at outgoing edges
		return ALL_EDGES;
	}
	
	gather_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const
	{
		cout << "gather(), edge=" << edge.source().id() << "->" << edge.target().id() << ", called from vid=" << vertex.id() << endl;
		
		gather_type gathered;
		
		// add id of other vertex of edge and add id->beta to map if message source
		if (edge.target().id()==vertex.id())
		{
			// incoming edge, outgoing message, only if target is non-observed
			if (!edge.source().data().is_observed)
			{
				gathered.message_targets.insert(edge.source().id());
				cout << "added " << edge.source().id() << " as message target" << endl;
			}
		}
		else
		{
			// outgoing edge, incoming message with beta
			gathered.message_source_betas[edge.target().id()]=edge.data().beta;
			cout << "added " << edge.target().id() << " as message source" << endl;	
		}
					
		cout << "gathered=" << gathered << endl;
		
		return gathered;
	}

	void apply(icontext_type& context, vertex_type& vertex, const gather_type& total)
	{
		cout << "apply(), vid=" << vertex.id() << endl;
		cout << "total=" << total << endl;

		// reset incoming messages
		vertex.data().multiplied_incoming_messages.clear();

		// iterate over message targets
		for (set<vertex_id_type>::const_iterator target_it=total.message_targets.begin(); target_it!=total.message_targets.end(); ++target_it)
		{
			vertex_id_type target_id=*target_it;
		
			// iterate over message sources (and the betas)
			for (map<vertex_id_type, VectorXd>::const_iterator source_it=total.message_source_betas.begin(); source_it!=total.message_source_betas.end(); ++source_it)
			{
				vertex_id_type source_id=source_it->first;
				
				// we dont need to consider the kernel matrix of a edge with itself since this is always omitted in the message scheduling
				if (source_id==target_id)
					continue;
					
				cout << "adding message from " << source_id << " to " << vertex.id() << " to construct message from " << vertex.id() << " to " << target_id << endl;
			
				// extract K and beta for incoming message
				MatrixXd K=vertex.data().kernel_dict[pair<vertex_id_type, vertex_id_type>(target_id,source_id)];
				VectorXd beta=source_it->second;
			
				// if beta has zero rows, it is initialised to constant with unit norm (first iteration)
				if (!beta.rows())
				{
					beta=VectorXd::Constant(K.cols(), 1.0);
					beta=beta/beta.norm();
				}
			
				cout << "K_" << vertex.id() << "^(" << target_id << "," << source_id << "):" << endl << K << endl;
				cout << "times" << endl;
				cout << "beta_(" << source_id << "," << vertex.id() << "):" << endl << beta << endl;
			
				// for a fixed source and target node, compute incoming kernelbp message
				VectorXd message=K*beta;
				cout << "message: " << message << endl;
				
				// multiply all messages together
				if (vertex.data().multiplied_incoming_messages.find(target_id)==vertex.data().multiplied_incoming_messages.end())
					vertex.data().multiplied_incoming_messages[target_id]=message;
				else
				{
					cout << "old message product: " << vertex.data().multiplied_incoming_messages[target_id] << endl;
					vertex.data().multiplied_incoming_messages[target_id]=vertex.data().multiplied_incoming_messages[target_id].cwiseProduct(message);
				}
				
				cout << "new message product: " << vertex.data().multiplied_incoming_messages[target_id] << endl;
			}
		}
	}
	
	edge_dir_type scatter_edges(icontext_type& context,	const vertex_type& vertex) const
	{
		// nodes behind incoming edges are the ones to which messages are sent
		return IN_EDGES;
	}
	
	void scatter(icontext_type& context, const vertex_type& vertex, edge_type& edge) const
	{
		cout << "scatter(), edge=" << edge.source().id() << "->" << edge.target().id() << ", called from vid=" << vertex.id() << endl;
		cout << "computing message from vid=" << vertex.id() << " to vid=" << edge.source().id() << endl;

		vertex_id_type message_target=edge.source().id();
		
		// find out whether full rank or incomplete Cholesky mode

		// distinguish case this node being observed or not
		VectorXd new_beta;
		if (edge.target().data().is_observed)
		{
			cout << "observed target" << endl;
			
			// extract system solutions and observation kernel vector, base on full rank or incomplete Cholesky
			if (edge.data().full_rank)
			{
				cout << "full rank case" << endl;
				
				MatrixXd L_s=edge.data().solution_matrices["L_s"];
				cout << "L_s:" << L_s << endl;
				MatrixXd L_t=edge.data().solution_matrices["L_t"];
				cout << "L_t:" << L_t << endl;
				VectorXd k=vertex.data().kernel_dict_obs.at(message_target);
				cout << "k:" << k << endl;
			
				// L_{s}^{-T}(L_{s}^{-1}(L_{t}^{-T}(L_{t}^{-1}k_{t}^{s}), from right to left, 4 solver calls
				new_beta=k;
				new_beta=L_t.triangularView<Lower>().solve(new_beta);
				new_beta=L_t.transpose().triangularView<Upper>().solve(new_beta);
				new_beta=L_s.triangularView<Lower>().solve(new_beta);
				new_beta=L_s.transpose().triangularView<Upper>().solve(new_beta);
			}
			else
			{
				cout << "incomplete Cholesky case" << endl;
				MatrixXd Q_s=edge.data().solution_matrices["Q_s"];
				cout << "Q_s:" << Q_s << endl;
				MatrixXd R_s=edge.data().solution_matrices["R_s"];
				cout << "R_s:" << R_s << endl;
				MatrixXd P_s=edge.data().solution_matrices["P_s"];
				cout << "P_s:" << P_s << endl;
				
				MatrixXd Q_t=edge.data().solution_matrices["Q_t"];
				cout << "Q_t:" << Q_t << endl;
				MatrixXd R_t=edge.data().solution_matrices["R_t"];
				cout << "R_t:" << R_t << endl;
				MatrixXd P_t=edge.data().solution_matrices["P_t"];
				cout << "P_t:" << P_t << endl;
				
				MatrixXd W=edge.data().solution_matrices["W"];
				cout << "W:" << W << endl;
				
				VectorXd k=vertex.data().kernel_dict_obs.at(message_target);
				cout << "k:" << k << endl;
				
				// R_{s}^{-1}(Q_{s}^{T}((P_{s}(W_{s}W_{t}^{T}))(R_{t}^{-1}(Q_{t}^{T}(P_{t}k_{\mathcal{I}_{t}}^{(s)})))
				new_beta=k;
				new_beta=P_t.transpose()*new_beta;
				new_beta=Q_t.transpose()*new_beta;
				new_beta=R_t.triangularView<Upper>().solve(new_beta);
				new_beta=W*new_beta;
				new_beta=P_s.transpose()*new_beta;
				new_beta=Q_s.transpose()*new_beta;
				new_beta=R_s.triangularView<Upper>().solve(new_beta);
			}
		}
		else
		{
			cout << "non-observed target" << endl;
			cout << "multiplied_incoming_messages: " << vertex.data().multiplied_incoming_messages << endl;
			
			// extract system solutions, depending on full rank or incomplete Cholesky
			if (edge.data().full_rank)
			{
				cout << "full rank case" << endl;
				MatrixXd L_s=edge.data().solution_matrices["L_s"];
				cout << "L_s:" << L_s << endl;
		
				VectorXd k;
				if (!vertex.data().multiplied_incoming_messages.size())
				{
					cout << "no incoming messages, using constant unit norm vector" << endl;
					k=VectorXd::Constant(L_s.cols(), 1.0/sqrt(L_s.cols()));
				}
				else
				{
					k=vertex.data().multiplied_incoming_messages.at(message_target);
				}
				cout << "k:" << k << endl;
				
				// (K_{s}+\lambda I){}^{-1}k_{ut}^{(s)}=L_{s}^{-T}(L_{s}^{-1}k_{ut}^{(s)}) from right to left, 2 solver calls
				new_beta=k;
				new_beta=L_s.triangularView<Lower>().solve(new_beta);
				new_beta=L_s.transpose().triangularView<Upper>().solve(new_beta);
			}
			else
			{
				cout << "incomplete Cholesky case" << endl;
				
				MatrixXd Q_s=edge.data().solution_matrices["Q_s"];
				cout << "Q_s:" << Q_s << endl;
				MatrixXd R_s=edge.data().solution_matrices["R_s"];
				cout << "R_s:" << R_s << endl;
				MatrixXd P_s=edge.data().solution_matrices["P_s"];
				cout << "P_s:" << P_s << endl;
				
				MatrixXd W=edge.data().solution_matrices["W"];
				cout << "W:" << W << endl;
				
				VectorXd k;
				if (!vertex.data().multiplied_incoming_messages.size())
				{
					cout << "no incoming messages, using constant unit norm vector" << endl;
					k=VectorXd::Constant(W.cols(), 1.0/sqrt(W.cols()));
				}
				else
				{
					k=vertex.data().multiplied_incoming_messages.at(message_target);
				}
				cout << "k:" << k << endl;
				
				// R_{s}^{-1}(Q_{s}^{T}(P_{s}^{T}k_{t}^{(s)}))
				new_beta=k;
				new_beta=W*new_beta;
				new_beta=P_s.transpose()*new_beta;
				new_beta=Q_s.transpose()*new_beta;
				new_beta=R_s.triangularView<Upper>().solve(new_beta);
			}
		}

		// normalise
		new_beta=new_beta/new_beta.norm();
		
		// check whether has changed or not yet existed
		double difference;
		if (!edge.data().beta.rows())
			difference=numeric_limits<double>::infinity();
		else
			difference=(new_beta-edge.data().beta).norm();

		cout << "beta norm difference is " << difference << endl;
		if (difference>BETA_EPSILON)
		{
			// store new message and signal depending node if beta has changed or has not yet existed
			edge.data().beta=new_beta;
			context.signal(edge.source());
			cout << "beta has changed, new_beta=" << new_beta << "\nhas norm=" << new_beta.norm() << ", signalling vid=" << edge.source().id() << endl;
		}
		else
		{
			cout << "converged!\n";
		}
		
		cout << "beta: " << edge.source().id() << "->" << edge.target().id() << ": " << edge.data().beta << endl;
	}
};

#endif //KERNELBP_PROGRAM
