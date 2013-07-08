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
	ifstream infile;
	infile.open(filename.c_str());
	
	if (infile.fail())
	{
		cout << "file \"" << filename << "\" does not exist!" << endl;
	}

	// first, count rows, columns
	string line;
	rows=0;
	cols=0;
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

MatrixXd load_ascii_matrix(string filename)
{
	// load size
	unsigned rows, cols;
	get_ascii_matrix_size(filename, rows, cols);
	
	// allocate memory
	MatrixXd matrix(rows, cols);
	
	// parse matrix
	ifstream infile;
	infile.open(filename.c_str());
	
	if (infile.fail())
	{
		cout << "file \"" << filename << "\" does not exist!" << endl;
	}

	string line;
	for (unsigned i=0; i<rows; ++i)
	{
		getline(infile, line);
		stringstream strm(line);
		for (unsigned j=0; j<cols; ++j)
			strm >> matrix(i, j);
	}
	infile.close();
	
	return matrix;
}

void save_ascii_matrix(MatrixXd matrix, string filename)
{
	ofstream outfile;
	outfile.open(filename.c_str());
	
	if (outfile.fail())
	{
		cout << "file \"" << filename << "\" could not be opened/created!" << endl;
	}

	for (unsigned i=0; i<matrix.rows(); ++i)
	{
		for (unsigned j=0; j<matrix.cols(); ++j)
			outfile << matrix(i, j);
			
		outfile << "\n";
	}
	
	outfile.close();
}

#endif //ASCII_MATRIX_LOADER_H
