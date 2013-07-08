#ifndef STREAM_OPERATORS_H
#define STREAM_OPERATORS_H

#include <graphlab.hpp>
using namespace graphlab;

#include <Eigen/Dense>
using namespace Eigen;

#include <sstream>
#include <list>
#include <map>
using namespace std;

oarchive& operator <<(oarchive& oarc, const MatrixXd& m) 
{
	oarc << m.rows();
	oarc << m.cols();
	for (unsigned i=0; i<m.rows(); ++i)
	{
		for (unsigned j=0; j<m.cols(); ++j)
		{
			oarc << m(i, j);
		}
	}
	return oarc;
}

oarchive& operator <<(oarchive& oarc, const VectorXd& m) 
{
	oarc << m.rows();
	for (unsigned i=0; i<m.rows(); ++i)
	{
		oarc << m[i];
	}
	return oarc;
}

iarchive& operator >>(iarchive &iarc, MatrixXd &m)
{
	long rows;
	long cols;
	iarc >> rows;
	iarc >> cols;
	m=MatrixXd(rows, cols);
	for (unsigned i=0; i<rows; ++i)
	{
		for (unsigned j=0; j<cols; ++j)
		{
			iarc >> m(i, j);
		}
	}
	return iarc;
}

iarchive& operator >>(iarchive &iarc, VectorXd &v)
{
	long size;
	iarc >> size;
	v=VectorXd(size);
	for (unsigned i=0; i<size; ++i)
	{
		iarc >> v[i];
	}
	return iarc;
}

stringstream& operator >>(stringstream &strm, MatrixXd &m)
{
	long rows;
	long cols;
	strm >> rows;
	strm >> cols;
	m=MatrixXd(rows, cols);
	for (unsigned i=0; i<rows; ++i)
	{
		for (unsigned j=0; j<cols; ++j)
		{
			strm >> m(i, j);
		}
	}
	return strm;
}

stringstream& operator >>(stringstream &strm, VectorXd &v)
{
	long size;
	strm >> size;
	v=VectorXd(size);
	for (unsigned i=0; i<size; ++i)
	{
		strm >> v[i];
	}
	return strm;
}

oarchive& operator <<(oarchive& oarc, const map<vertex_id_type,MatrixXd>& m) 
{
	oarc << m.size();
	for (map<vertex_id_type,MatrixXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	{
		oarc << it->first;
		oarc << it->second;
	}

	return oarc;
}

oarchive& operator <<(oarchive& oarc, const map<vertex_id_type,VectorXd>& m) 
{
	oarc << m.size();
	for (map<vertex_id_type,VectorXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	{
		oarc << it->first;
		oarc << it->second;
	}

	return oarc;
}

iarchive& operator >>(iarchive &iarc, map<vertex_id_type,MatrixXd> &m)
{
	m.clear();
	size_t size;
	iarc >> size;
	for (size_t i=0; i<size; ++i)
	{
		vertex_id_type id;
		MatrixXd v;
		iarc >> id;
		iarc >> v;
		m[id]=v;
	}
	
	return iarc;
}

iarchive& operator >>(iarchive &iarc, map<vertex_id_type,VectorXd> &m)
{
	m.clear();
	size_t size;
	iarc >> size;
	for (size_t i=0; i<size; ++i)
	{
		vertex_id_type id;
		VectorXd v;
		iarc >> id;
		iarc >> v;
		m[id]=v;
	}
	
	return iarc;
}

oarchive& operator <<(oarchive& oarc, const map<pair<vertex_id_type,vertex_id_type>, MatrixXd> m) 
{
	oarc << m.size();
	for (map<pair<vertex_id_type,vertex_id_type>, MatrixXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	{
		oarc << it->first;
		oarc << it->second;
	}

	return oarc;
}

iarchive& operator >>(iarchive &iarc, map<pair<vertex_id_type,vertex_id_type>, MatrixXd> &m)
{
	m.clear();
	size_t size;
	iarc >> size;
	for (size_t i=0; i<size; ++i)
	{
		pair<vertex_id_type,vertex_id_type> ids;
		MatrixXd mat;
		iarc >> ids;
		iarc >> mat;
		m[ids]=mat;
	}
	
	return iarc;
}

oarchive& operator <<(oarchive& oarc, const map<pair<vertex_id_type,vertex_id_type>, VectorXd> m) 
{
	oarc << m.size();
	for (map<pair<vertex_id_type,vertex_id_type>, VectorXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	{
		oarc << it->first;
		oarc << it->second;
	}

	return oarc;
}

iarchive& operator >>(iarchive &iarc, map<pair<vertex_id_type,vertex_id_type>, VectorXd> &m)
{
	m.clear();
	size_t size;
	iarc >> size;
	for (size_t i=0; i<size; ++i)
	{
		pair<vertex_id_type,vertex_id_type> ids;
		VectorXd vec;
		iarc >> ids;
		iarc >> vec;
		m[ids]=vec;
	}
	
	return iarc;
}

oarchive& operator <<(oarchive& oarc, const map<string,MatrixXd>& m) 
{
	oarc << m.size();
	for (map<string,MatrixXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	{
		oarc << it->first;
		oarc << it->second;
	}

	return oarc;
}

iarchive& operator >>(iarchive &iarc, map<string,MatrixXd> &m)
{
	m.clear();
	size_t size;
	iarc >> size;
	for (size_t i=0; i<size; ++i)
	{
		string key;
		MatrixXd mat;
		iarc >> key;
		iarc >> mat;
		m[key]=mat;
	}
	
	return iarc;
}

oarchive& operator <<(oarchive& oarc, const vector<vertex_id_type>& v) 
{
	oarc << v.size();
	for (vector<vertex_id_type>::const_iterator it=v.begin(); it!=v.end(); ++it)
	{
		oarc << *it;
	}

	return oarc;
}

iarchive& operator >>(iarchive &iarc, vector<VectorXd> &v)
{
	v.clear();
	size_t size;
	iarc >> size;
	for (size_t i=0; i<size; ++i)
	{
		VectorXd vec;
		iarc >> vec;
		v.push_back(vec);
	}
	
	return iarc;
}

ostream& operator <<(ostream& out, const set<vertex_id_type>& s) 
{
	for (set<vertex_id_type>::const_iterator it=s.begin(); it!=s.end(); ++it)
	 	out << *it << ", ";
	 	
	return out;
}

ostream& operator <<(ostream& out, const map<vertex_id_type,VectorXd>& m) 
{
	for (map<vertex_id_type,VectorXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	 	out << it->first << "->" << it->second << endl;
	 	
	return out;
}

ostream& operator <<(ostream& out, const map<vertex_id_type, MatrixXd>& m) 
{
	for (map<vertex_id_type, MatrixXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	 	out << it->first << "->" << it->second << endl;
	 	
	return out;
}

ostream& operator <<(ostream& out, const map<vertex_id_type, map<vertex_id_type,MatrixXd> >& m) 
{
	for (map<vertex_id_type, map<vertex_id_type,MatrixXd> >::const_iterator it=m.begin(); it!=m.end(); ++it)
	 	out << it->first << "->" << it->second << endl;
	 	
	return out;
}

ostream& operator <<(ostream& out, const pair<vertex_id_type,vertex_id_type>& pair) 
{
	out << "(" << pair.first << "," << pair.second << ")";	 	
	return out;
}

ostream& operator <<(ostream& out, const map<pair<vertex_id_type,vertex_id_type>, MatrixXd>& m) 
{
	for (map<pair<vertex_id_type,vertex_id_type>, MatrixXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	 	out << it->first << "->" << it->second << endl;
	 	
	return out;
}

ostream& operator <<(ostream& out, const map<pair<vertex_id_type,vertex_id_type>, VectorXd>& m) 
{
	for (map<pair<vertex_id_type,vertex_id_type>, VectorXd>::const_iterator it=m.begin(); it!=m.end(); ++it)
	 	out << it->first << "->" << it->second << endl;
	 	
	return out;
}
#endif //STREAM_OPERATORS_H
