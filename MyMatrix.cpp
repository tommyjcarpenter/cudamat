/*Copyright (c) 2013 Tommy Carpenter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "MyMatrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <time.h>
#include <exception>

#define BLOCKSIZE 512

using namespace std;

// constructor for MyMatrix, a matrix with padding for GPU purposes
MyMatrix::MyMatrix(int new_rows, int new_cols, int padrr, int padcc)
{
	rows = new_rows;
	cols = new_cols;
	padr = padrr;
	padc = padcc;
	try
	{
		data = new float[new_rows*new_cols];
	}
	catch (bad_alloc&)
    {
		cout << "Exception in memory allocation. Could not allocate enough memory for the array." << endl << flush;
		cout << "Exception in array memory allocation. Could not allocate enough memory for the array." << endl << flush;
		exit(1);
	}
	 catch (exception& e)
     {
       cout << "Some other exception was caught (not bad_alloc) " << e.what() << endl << flush;
		 exit(1);
     }
}

// destructor
MyMatrix::~MyMatrix(void)
{
	delete data; // delete the data array
}


// print a matrix, used for debugging
void MyMatrix::printData()
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << " " << (data)[i*cols + j];
		}
		cout << endl;
	}
}

// reads a matrix from a textfile and returns a pointer to a new MyMatrix object that
// has been dynamically allocated to the heap. This function pads the matrix for GPU calculations.
// It will be output normally (depadded) later
// input: filename to read from
// output: MyMatrix object pointer
// side effects: none
MyMatrix *MyMatrix::readMatrix(string filename)
{
   ifstream myReadFile;
   myReadFile.open(filename.c_str());
	
	int r, c;
   myReadFile >> r >> c;

	// we need to pad the matrix so that the size is a multiple of blocksize. 
	int padr = 0;
	int padc = 0;

	// TO DO LIST: MAKE BLOCKSIZE A PARAMATER ONCE FIGURED OUT HOW TO MAKE BLOCK SIZE DYNAMIC.
	while((r + padr) % BLOCKSIZE != 0) padr++;
	while((c + padc) % BLOCKSIZE != 0) padc++;

	int totalr = padr + r;
	int totalc = padc + c;

   // dynamically allocate a new mymatrix object
	MyMatrix *newmat = new MyMatrix(totalr, totalc, padr, padc);
	int index;
	
   //cout << "pritning r: " << r << "printing c: " << c << "totalc:" << totalc << " total r: " << totalr << endl;   
	for (int i = 0; i < totalr; i++)
	{   
		for (int j = 0; j < totalc; j++)
		{ 
			index = i*totalc + j;
			if ((j < c) && (i < r))
			{  
		      myReadFile >> (*newmat).data[index];
			}
			else
         {
	   		(*newmat).data[index] = 0;	
         }
		}
    }
	myReadFile.close();
	return newmat;
}

// takes in rows and columns and generates a random matrix. Returns a pointer to that dynamically allocated matrix. 
MyMatrix *MyMatrix::generateRandomMatrix(int r, int c)
{
   // we need to pad the matrix so that the size is a multiple of BLOCKSIZE. 
	int padr = 0;
	int padc = 0;

	while((r + padr) % BLOCKSIZE != 0) padr++;
	while((c + padc) % BLOCKSIZE != 0) padc++;

	int totalr = padr + r;
	int totalc = padc + c;
   int index;

   // dynamically allocate a new MyMatrix to the heap
	MyMatrix *newmat = new MyMatrix(totalr, totalc, padr, padc);
   
   for (int i = 0; i < totalr; i++)
	{   
		for (int j = 0; j < totalc; j++)
		{ 
			index = i*totalc + j;
			if ((j < c) && (i < r))
			{  
		      (*newmat).data[index] = (float)rand()/(float)RAND_MAX;
			}
			else
         {
	   		(*newmat).data[index] = 0;	
         }
		}
    }
	return newmat;
}

// writes a matrix to a textfile. Depads the matrix (padding is used for internal representation)
void MyMatrix::writeMatrix(string filename)
{
  ofstream MatFile;
  MatFile.open(filename.c_str());
  
  MatFile << rows-padr << " " << cols-padc << endl;
  for (int i = 0; i < rows; i++)
  {
	  for (int j = 0; j < cols; j++)
	  {
		   if ((j < cols-padc) && (i < rows-padr))
			{  
		         MatFile << setprecision (9) << data[i*cols + j] << "  ";
			}
	  }
	  MatFile << endl;
  }
  MatFile.close();
}

// I was using this for debugging
void MyMatrix::writeDifference(MyMatrix &Mat1, MyMatrix &Mat2, string filename)
{
	MyMatrix difference (Mat1.rows, Mat1.cols, Mat1.padr, Mat1.padc);
    for (int i = 0; i < Mat1.rows*Mat1.cols; i++)
     {      difference.data[i] = Mat1.data[i] - Mat2.data[i];
          if ((difference.data[i] < .01) && (difference.data[i] > -.01))
             difference.data[i] = 0;
     }
	difference.writeMatrix(filename);
}

// multiples two matrices, writes result to file. 
void MyMatrix::multMats(string filename1, string filename2, string gpuoutfname, int genNew, int n, int p, int m)
{
   MyMatrix *Mat1;
   MyMatrix *Mat2;
   
	if (genNew == 1)
	{	   
		Mat1 = MyMatrix::generateRandomMatrix(n, p);
	   Mat2 = MyMatrix::generateRandomMatrix(p, m);
	}
   else
   {
	   cout << "Reading matrices from file..." << endl;
	   Mat1 = MyMatrix::readMatrix(filename1);
	   Mat2 = MyMatrix::readMatrix(filename2);
   }

	cout << "Multiplying... " << endl;

   // make the call
	MyMatrix result = result.CUDAMatMatMultiply(*Mat1, *Mat2);

   cout << "Writing output file..." << endl;
	result.writeMatrix(gpuoutfname);

   if (genNew)
   {
      cout << "Writing the random input matrix to a file..." << endl;
      (*Mat1).writeMatrix(filename1);
      (*Mat2).writeMatrix(filename2);
   }

   cout << "Done!" << endl<< flush;
   delete Mat1;
   delete Mat2;
}


// reads in one matrix, raises it to a given power, writes result to file. 
void MyMatrix::raisePowerOf2(string filename1, string gpuoutfname, int genNew, int Times, int n)
{ 
   MyMatrix *Mat1;

	if (genNew == 1)
		Mat1 = MyMatrix::generateRandomMatrix(n, n);
   else
   {
	   cout << "Reading matrix from file..." << endl;
	   Mat1 = MyMatrix::readMatrix(filename1);
   }

	cout << "Multiplying..." << endl;
	
   // MAKE THE CALL
   MyMatrix result = result.CUDAMatPower(*Mat1, Times);
   
	cout << "Writing output file..." << endl;
	result.writeMatrix(gpuoutfname);

   if (genNew)
   {
      cout << "Writing the random input matrix to a file..." << endl;
      (*Mat1).writeMatrix(filename1);
   }

   cout << "Done!" <<endl;	
}



