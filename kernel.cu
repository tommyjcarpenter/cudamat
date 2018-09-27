#include "MyMatrix.h"
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// multiplies two matrices and returns a new matrix
//intput two mymatrix objects
// output mymatrix
MyMatrix MyMatrix::CUDAMatMatMultiply(MyMatrix *Mat1, MyMatrix *Mat2)
{
     // create cuda events for timing
     cudaError_t cudaStatus;
     float elapsedTimeExecution;
     cudaEvent_t startExec, stopExec;
     cudaEventCreate(&startExec);
     cudaEventCreate(&stopExec);

    // various paramaters we need
     dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
     int Arows = Mat1->rows;
     int Acols = Mat1->cols;
     int Bcols = Mat2->cols;
     dim3 dimGrid(ceil((double) Bcols/BLOCKSIZE), ceil((double) Arows/BLOCKSIZE));

     // ptrs to input matrices data
     double *ptr1 = Mat1->data;
     double *ptr2 = Mat2->data;

     // Output matrix
     MyMatrix OutputMat(Arows, Bcols, Mat1->padr, Mat2->padc);
     // Pointer to output matrix; cuda will copy to this
     double *outmatptr = OutputMat.data;

     // ready to preform a kernel; record that this even is happeneing
     cudaStatus = cudaEventRecord(startExec, 0);
     if (cudaStatus != cudaSuccess){ fprintf(stderr, "event record failure!"); goto Error;}

     // CUDA KERNEL CALL
     MatrixMulKernel<<< dimGrid, dimBlock>>>(outmatptr, ptr1, ptr2, Arows, Acols, Bcols);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

     // find the time the execution took
     cudaEventRecord(stopExec, 0);
     cudaEventSynchronize(stopExec);
     if (cudaStatus != cudaSuccess) {fprintf(stderr, "event record failure!"); goto Error; }
     cudaStatus = cudaEventElapsedTime(&elapsedTimeExecution, startExec, stopExec);
     if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaEventElapsedTime returned error code %d!\n", cudaStatus); goto Error;}
     cout << "Using Cuda Timers, the total kernel execution time was  " << elapsedTimeExecution << "ms" << endl;

Error:
    // either we have errd, or the program finished natrually.
    // either way, free all device memory useage!
    cudaEventDestroy(startExec);
    cudaEventDestroy(stopExec);
    return OutputMat;
}

// Raise a Matrix to a power using CUDA
// intput, mymatrix, int times
// output, mymatrix
MyMatrix MyMatrix::CUDAMatPower(MyMatrix *Mat1, int TIMES)
{
     // create cuda events for timing
     cudaError_t cudaStatus;
     float elapsedTimeExecution;
     cudaEvent_t startExec, stopExec;
     cudaEventCreate(&startExec);
     cudaEventCreate(&stopExec);

     // block and thread size stuff
     dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
     int width = Mat1->rows;
     dim3 dimGrid(ceil((double) width/BLOCKSIZE), ceil((double) width/BLOCKSIZE));
     int matsize = width*width*sizeof(double);

     // GPU device pointers
     double *GPUTempMat;

     // Output matrix
     MyMatrix OutputMat(width, width, Mat1->padr, Mat1->padc);

     // pointers to matrix data elements
     double *outmatptr = OutputMat.data; // pass this on all subsequent squares

     // ready to preform a kernel; record that this even is happeneing
     cudaStatus = cudaEventRecord(startExec, 0);
     if (cudaStatus != cudaSuccess){    fprintf(stderr, "event record failure!"); goto Error;}

     // Allocate GPU buffers for three vectors in GPU global Memory
     cudaStatus = cudaMalloc((void**)&GPUTempMat, matsize);
     if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!");goto Error;}

     // keep squaring until the total number is greater than the orginal number asked for
     for (double T = 2; T <= TIMES; T *=2)
     {
         // on the first time pass in the matrix data so it can be squared
         if (T == 2){
            MatrixMulKernel<<< dimGrid, dimBlock>>>(outmatptr, Mat1->data, Mat1->data, width, width, width);
         }
         else{
            cudaStatus = cudaMemcpy(GPUTempMat, outmatptr, matsize, cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!");    goto Error; }
            MatrixMulKernel<<< dimGrid, dimBlock>>>(outmatptr, GPUTempMat, GPUTempMat, width, width, width);
         }

         // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
         cudaStatus = cudaDeviceSynchronize();
         if (cudaStatus != cudaSuccess)  {fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus); goto Error;}

      }

    // find the time the execution took
    cudaEventRecord(stopExec, 0);
    cudaEventSynchronize(stopExec);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "event record failure!"); goto Error; }
    cudaStatus = cudaEventElapsedTime(&elapsedTimeExecution, startExec, stopExec);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaEventElapsedTime returned error code %d!\n", cudaStatus); goto Error;}
    cout << "Using Cuda Timers, the total kernel execution time was  " << elapsedTimeExecution << "ms" << endl;

Error:
    // either we have errd, or the program finished natrually.
    // either way, free all device memory useage!
    cudaEventDestroy(startExec);
    cudaEventDestroy(stopExec);
    cudaFree(GPUTempMat);

    return OutputMat;
}



__global__ void addKernel(double *c, const double *a, const double *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// multiplies two matrices and outputs a new matrix
// input: two mymatrix data doubles
// output none really, mymatrix output double
__global__ void MatrixMulKernel(double *OutMat, double *Mat1, double *Mat2,  int Arows, int Acols, int Bcols)
{
    // row and column within submatrix
    int blockrow =  blockIdx.y;//*
    int row = threadIdx.y;
    int blockcol = blockIdx.x;
    int col =  threadIdx.x ;

    // allocate these arrays only once we can change the values in them later
    __shared__ double subAshared[BLOCKSIZE*BLOCKSIZE];
    __shared__ double subBshared[BLOCKSIZE*BLOCKSIZE];
    double Cvalue=0;

    for (int B = 0; B < ceil((double)(Acols / BLOCKSIZE)) + 1; B++)
    {
        // fetch from global memory
        // yes, these took a LONG time to figure out. Pencil and Paper FTW!
        int Mat1index = (row + blockrow*BLOCKSIZE)*Acols + col + B*BLOCKSIZE;
        int Mat2index = (B*BLOCKSIZE + row)*Bcols + BLOCKSIZE*blockcol + col;

        if (Mat1index < Arows*Acols)
            subAshared[row*BLOCKSIZE + col] = Mat1[Mat1index];
        else
            subAshared[row*BLOCKSIZE + col] = 0;

        if (Mat2index < Acols*Bcols)
            subBshared[row*BLOCKSIZE + col] = Mat2[Mat2index];
        else
            subBshared[row*BLOCKSIZE + col] = 0;

        __syncthreads();

        for (int j = 0; j < BLOCKSIZE; j++)
        {
            if ((row*BLOCKSIZE + j < BLOCKSIZE*BLOCKSIZE) && (j*BLOCKSIZE + col < BLOCKSIZE*BLOCKSIZE))
            {
                Cvalue += subAshared[row*BLOCKSIZE + j]*subBshared[j*BLOCKSIZE + col];
            }
        }

        __syncthreads();

    }
    if ((row < Arows) && (col < Bcols))
    {
        int finalmatrow = blockrow*BLOCKSIZE + row;
        int finalmatcol = blockcol*BLOCKSIZE + col;
        OutMat[finalmatrow*Bcols +  finalmatcol] = Cvalue;
    }
}
