#ifndef ASCII_MATRIX_LOADER_H
#define ASCII_MATRIX_LOADER_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

void get_ascii_matrix_size(string filename, unsigned& rows, unsigned& cols)
{
	rows=0;
	cols=0;

	ifstream infile;
	infile.open(filename.c_str());
	
	if (infile.fail())
	{
		logstream(LOG_ERROR) << "Error opening file \"" << filename << "\"." << endl;
	}
	else
	{
		// first, count rows, columns
		string line;
		while (getline(infile, line))
		{
			if (!rows)
			{
				stringstream strm(line);
				string temp;
				while (strm >> temp)
				{
					cols++;
				}
			}

			rows++;
		}
		infile.close();
	}
}

MatrixXd load_ascii_matrix(string filename, string prefix="")
{
	string full_filename;
	full_filename.append(prefix);
	full_filename.append(filename);

	// load size
	unsigned rows, cols;
	get_ascii_matrix_size(full_filename, rows, cols);
	
	// allocate memory
	MatrixXd matrix(rows, cols);
	
	// parse matrix
	ifstream infile;
	infile.open(full_filename.c_str());
	
	if (infile.fail())
	{
		logstream(LOG_ERROR) << "Error opening file \"" << full_filename << "\"." << endl;
		
	}
	else
	{
		string line;
		for (unsigned i=0; i<rows; ++i)
		{
			getline(infile, line);
			stringstream strm(line);
			for (unsigned j=0; j<cols; ++j)
				strm >> matrix(i, j);
		}
	}
	infile.close();
	
	return matrix;
}

void save_ascii_matrix(MatrixXd matrix, string filename, string prefix="")
{
	string full_filename;
	full_filename.append(prefix);
	full_filename.append(filename);
	
	ofstream outfile;
	outfile.open(full_filename.c_str());
	
	if (outfile.fail())
	{
		logstream(LOG_ERROR) << "Error writing to file \"" << full_filename << "\"." << endl;
	}
	else
	{
		for (unsigned i=0; i<matrix.rows(); ++i)
		{
			for (unsigned j=0; j<matrix.cols(); ++j)
				outfile << matrix(i, j);
			
			outfile << endl;
		}
	
	}
	
	outfile.close();
}

#endif //ASCII_MATRIX_LOADER_H
