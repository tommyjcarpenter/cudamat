#include "MyMatrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <time.h>
#include <exception>
#include "cuda_runtime.h"

#define BLOCKSIZE 512

using namespace std;

// constructor for MyMatrix, a matrix with padding for GPU purposes
MyMatrix::MyMatrix(int new_rows, int new_cols, int padrr, int padcc)
{
    rows = new_rows;
    cols = new_cols;
    padr = padrr;
    padc = padcc;

    cout << "allocating memory:" << new_rows*new_cols*sizeof(double) << "bytes" << endl;
    cudaError_t cudaStatus = cudaMallocManaged(&data, new_rows*new_cols*sizeof(double));
    if (cudaStatus != cudaSuccess){
        cout << cudaStatus << endl << flush;
        exit(1);
    }
    cudaDeviceSynchronize();
    cout << "allocating memory successful:" << cudaStatus << endl;
}

// destructor
MyMatrix::~MyMatrix(void)
{
    cudaFree(data); // delete the data array
}

double MyMatrix::getrc(int r, int c)
{
    return data[r*cols+c];
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

    cout << "generating random matrix" << endl;

    // dynamically allocate a new MyMatrix to the heap
    MyMatrix *newmat = new MyMatrix(totalr, totalc, padr, padc);

    cout << "populating random matrix" << endl;

    for (int i = 0; i < totalr; i++)
        for (int j = 0; j < totalc; j++)
            (*newmat).data[i*totalc + j] = ( (j < c && i < r) ? Randomdouble(1.0, 1000.0) : 0);
    return newmat;
}

// writes a matrix to a textfile. Depads the matrix (padding is used for internal representation)
void MyMatrix::writeMatrix(string filename)
{
  ofstream MatFile;
  MatFile.open(filename.c_str());

  MatFile << rows-padr << " " << cols-padc << endl;
  for (int i = 0; i < rows-padr; i++)
  {
      for (int j = 0; j < cols-padc; j++){
           MatFile << setprecision (16) << data[i*cols + j] << " ";
      }
      MatFile << endl;
  }
  MatFile.close();
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
        cout << "Writing the random input matrix to a file..." << endl;
        (*Mat1).writeMatrix(filename1);
        (*Mat2).writeMatrix(filename2);
    }
    else
    {
        cout << "Reading matrices from file..." << endl;
        Mat1 = MyMatrix::readMatrix(filename1);
        Mat2 = MyMatrix::readMatrix(filename2);
    }

    // make the call
    cout << "Multiplying... " << endl;
    MyMatrix result = result.CUDAMatMatMultiply(*Mat1, *Mat2);
    cout << "Writing output file..." << endl;
    result.writeMatrix(gpuoutfname);

    // make the call
    cout << "Multiplying... " << endl;
    MyMatrix result9 = result.CUDAMatMatMultiply_cuda9(Mat1, Mat2);
    cout << "Writing output 9 file..." << endl;
    result9.writeMatrix("9"+gpuoutfname);

    // verify against cpu
    cout << "Verifying against CPU" << endl;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            double s = 0;
            for(int k = 0; k < p; k++) //p == M2->rows
                // essentially doing a dot product of ith row of M1 and jth column of M2
                s +=  Mat1->getrc(i,k) * Mat2->getrc(k,j);
            double difference = result9.getrc(i, j) - s;
            if (difference < -.001 || difference > .001) {
                cout << "MATCH FAILURE!" << " gpu=" << result9.data[i*Mat2->cols+j] << " cpu=" << s << endl;
            }
        }
    }

    cout << "Done!" << endl<< flush;
    delete Mat1;
    delete Mat2;
}


// reads in one matrix, raises it to a given power, writes result to file.
void MyMatrix::raisePowerOf2(string filename1, string gpuoutfname, int genNew, int Times, int n)
{
    MyMatrix *Mat1;

    if (genNew == 1) {
        Mat1 = MyMatrix::generateRandomMatrix(n, n);
         cout << "Writing the random input matrix to a file..." << endl;
        (*Mat1).writeMatrix(filename1);
    }
    else {
        cout << "Reading matrix from file..." << endl;
        Mat1 = MyMatrix::readMatrix(filename1);
    }

    cout << "Multiplying..." << endl;

    // MAKE THE CALL
    MyMatrix result = result.CUDAMatPower(*Mat1, Times);
    cout << "Writing output file..." << endl;
    result.writeMatrix(gpuoutfname);

    // MAKE THE CALL
    MyMatrix result9 = result.CUDAMatPower_cuda9(Mat1, Times);
    cout << "Writing output 9 file..." << endl;
    result9.writeMatrix("9"+gpuoutfname);

    cout << "Done!" <<endl;
}

// stolen from https://stackoverflow.com/questions/5289613/generate-random-float-between-two-floats
static double Randomdouble(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = b - a;
    double r = random * diff;
    return a + r;
}
