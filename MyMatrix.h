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

        double *data;
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
        MyMatrix CUDAMatMatMultiply_cuda9(MyMatrix *Mat1, MyMatrix *Mat2);
        MyMatrix CUDAMatPower( MyMatrix &Mat1, int times);
        MyMatrix CUDAMatPower_cuda9( MyMatrix *Mat1, int times);

        // gen new matrices
        static MyMatrix *readMatrix(string filename);
        static MyMatrix *generateRandomMatrix(int r, int c);
        static void multMats(string filename1, string filename2, string gpuoutfname, int genNew, int n, int p, int m);
        static void raisePowerOf2(string filename1, string gpuoutfname, int genNewm, int Times, int n);
};
