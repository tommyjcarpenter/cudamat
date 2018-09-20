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

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <valarray>
#include <iostream>

using namespace std;

class MyMatrix
{
	
public:
	MyMatrix(int new_rows, int new_cols, int padr, int padt);
	~MyMatrix(void);

	float *data; 
   int padr;
	int padc;
   int rows;
   int cols;

	// reading/writing functions
	void printData();
	void writeMatrix(string filename);
	void writeDifference(MyMatrix &Mat1, MyMatrix &Mat2, string filename);

   // cuda matrix operations
	MyMatrix CUDAMatMatMultiply(MyMatrix &Mat1, MyMatrix &Mat2);
	MyMatrix CUDAMatPower( MyMatrix &Mat1, int times);

   // gen new matrices
	static MyMatrix *readMatrix(string filename);
	static MyMatrix *generateRandomMatrix(int r, int c);
   static void multMats(string filename1, string filename2, string gpuoutfname, int genNew, int n, int p, int m);
	static void raisePowerOf2(string filename1, string gpuoutfname, int genNewm, int Times, int n);
};







