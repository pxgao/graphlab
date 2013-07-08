#ifndef GATHER_TYPE_H
#define GATHER_TYPE_H

#include <map>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

struct gather_type {
	// sets of kernelbp message sources and targets (oposite edge direction)
	map<vertex_id_type, VectorXd> message_source_betas;
	set<vertex_id_type> message_targets;
	
	gather_type() {}
	
	void save(oarchive& oarc) const
	{
		oarc << message_source_betas;
		oarc << message_targets;
	}
	
	void load(iarchive& iarc)
	{
		iarc >> message_source_betas;
		iarc >> message_targets;
	}
	
	gather_type& operator+=(const gather_type& other)
	{
		message_source_betas.insert(other.message_source_betas.begin(), other.message_source_betas.end());
		message_targets.insert(other.message_targets.begin(), other.message_targets.end());
		return *this;
	}
};

struct gather_type;
ostream& operator <<(ostream& out, const gather_type& gathered) 
{
	out << "gather_type:" << endl;
	out << "message_source_betas: " << gathered.message_source_betas << endl;
	out << "message_targets: " << gathered.message_targets;
	
	//out << *this << endl;
	return out;
}


#endif //GATHER_TYPE_H
