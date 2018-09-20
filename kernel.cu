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
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// hard code block size 
#define BLOCK_SIZE 16

// multiplies two matrices and returns a new matrix
//intput two mymatrix objects
// output mymatrix
MyMatrix MyMatrix::CUDAMatMatMultiply(MyMatrix &Mat1, MyMatrix &Mat2)
{   
	 // create cuda events for timing   
	 cudaError_t cudaStatus;
	 float elapsedTimeExecution;
	 cudaEvent_t startExec, stopExec;
	 cudaEventCreate(&startExec);
	 cudaEventCreate(&stopExec);

    // various paramaters we need
	 dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	 int Arows = Mat1.rows;
	 int Acols = Mat1.cols;
	 int Bcols = Mat2.cols;
	 dim3 dimGrid(ceil((float) Bcols/BLOCK_SIZE), ceil((float) Arows/BLOCK_SIZE));
    int finalmatsize = Arows*Bcols*sizeof(float);
	 int mat1size = Arows*Acols*sizeof(float);
	 int mat2size = Acols*Bcols*sizeof(float);

	 // ptrs to input matrices data
    float *ptr1 = &Mat1.data[0];
	 float *ptr2 = &Mat2.data[0];

	 // GPU device pointers
	 float *GPUOutMat, *GPUMat1, *GPUMat2;

	 // Output matrix
	 MyMatrix OutputMat(Arows, Bcols, Mat1.padr, Mat2.padc);
    // Pointer to output matrix; cuda will copy to this
	 float *outmatptr = &OutputMat.data[0];

	 // ready to preform a kernel; record that this even is happeneing
	 cudaStatus = cudaEventRecord(startExec, 0);
	 if (cudaStatus != cudaSuccess){	fprintf(stderr, "event record failure!"); goto Error;}

	 // Allocate GPU buffers for three vectors in GPU global Memory   
    cudaStatus = cudaMalloc((void**)&GPUOutMat, finalmatsize);
	 cudaStatus = cudaMalloc((void**)&GPUMat1, mat1size);
	 cudaStatus = cudaMalloc((void**)&GPUMat2, mat2size);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}


	 // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(GPUMat1, ptr1, mat1size, cudaMemcpyHostToDevice);
	 cudaStatus = cudaMemcpy(GPUMat2, ptr2, mat2size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
   
    // CUDA KERNEL CALL
	 MatrixMulKernel<<< dimGrid, dimBlock>>>(GPUOutMat, GPUMat1, GPUMat2, Arows, Acols, Bcols);

    // cudaThreadSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outmatptr, GPUOutMat, finalmatsize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

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
   cudaFree(GPUOutMat);
   cudaFree(GPUMat1);
   cudaFree(GPUMat2);
  
	return OutputMat;
}




// Raise a Matrix to a power using CUDA 
// intput, mymatrix, int times
// output, mymatrix
MyMatrix MyMatrix::CUDAMatPower(MyMatrix &Mat1, int TIMES)
{    
	 // create cuda events for timing
	 cudaError_t cudaStatus; 
	 float elapsedTimeExecution;
	 cudaEvent_t startExec, stopExec;
	 cudaEventCreate(&startExec);
	 cudaEventCreate(&stopExec);

	 // block and thread size stuff
	 dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	 int width = Mat1.rows;
	 dim3 dimGrid(ceil((float) width/BLOCK_SIZE), ceil((float) width/BLOCK_SIZE));
	 int matsize = width*width*sizeof(float);
	 
	 // GPU device pointers
	 float *GPUOutMat, *GPUTempMat;

	 // Output matrix
	 MyMatrix OutputMat(width, width, Mat1.padr, Mat1.padc);
      
	 // pointers to matrix data elements
    float *ptr1 = &Mat1.data[0]; // on the first iteration we pass this data
	 float *outmatptr = &OutputMat.data[0]; // pass this on all subsequent squares

    // ready to preform a kernel; record that this even is happeneing
	 cudaStatus = cudaEventRecord(startExec, 0);
	 if (cudaStatus != cudaSuccess){	fprintf(stderr, "event record failure!"); goto Error;}

  	 // Allocate GPU buffers for three vectors in GPU global Memory   
    cudaStatus = cudaMalloc((void**)&GPUOutMat, matsize);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!");goto Error;}    
	 cudaStatus = cudaMalloc((void**)&GPUTempMat, matsize);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!");goto Error;}
	
	 // keep squaring until the total number is greater than the orginal number asked for
	 for (float T = 2; T <= TIMES; T *=2)
    {	 // on the first time pass in the matrix data so it can be squared
       if (T == 2)
         cudaStatus = cudaMemcpy(GPUTempMat, ptr1, matsize, cudaMemcpyHostToDevice); // copy temp to GPU
       else
         cudaStatus = cudaMemcpy(GPUTempMat, outmatptr, matsize, cudaMemcpyHostToDevice); // outputmatrix already holds the last square
		
       // check for errors!
		 if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!");	goto Error;	}
       		
       // MULTIPLY TEMP * TEMP, PUT IN GPUOUTMAT
		 MatrixMulKernel<<< dimGrid, dimBlock>>>(GPUOutMat, GPUTempMat, GPUTempMat, width, width, width);

       // cudaThreadSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	    cudaStatus = cudaThreadSynchronize();
	    if (cudaStatus != cudaSuccess)  {fprintf(stderr, "cudaThreadSynchronize returned error code %d\n", cudaStatus);	goto Error;}

		 // TAKE THE RESULT AND TRANSFER BACK TO HOST probably a little optimization that can happen here, no need to transfe
		 // back to host every time but leaveing it like this for now
		 cudaStatus = cudaMemcpy(outmatptr, GPUOutMat, matsize, cudaMemcpyDeviceToHost);
		 if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error;	}
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
   cudaFree(GPUOutMat);
   cudaFree(GPUTempMat);

	return OutputMat;
}

__global__ void addKernel(float *c, const float *a, const float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// multiplies two matrices and outputs a new matrix
// input: two mymatrix data floats
// output none really, mymatrix output float
__global__ void MatrixMulKernel(float *OutMat, float *Mat1, float *Mat2,  int Arows, int Acols, int Bcols)
{	
	// row and column within submatrix
	int blockrow =  blockIdx.y;//*
	int row = threadIdx.y;
	int blockcol = blockIdx.x;
	int col =  threadIdx.x ;

	// allocate these arrays only once we can change the values in them later
	__shared__ float subAshared[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float subBshared[BLOCK_SIZE*BLOCK_SIZE];
	float Cvalue=0;

    for (int B = 0; B < ceil((float)(Acols / BLOCK_SIZE)) + 1; B++)
	{
		// fetch from global memory
		// yes, these took a LONG time to figure out. Pencil and Paper FTW!
		int Mat1index = (row + blockrow*BLOCK_SIZE)*Acols + col + B*BLOCK_SIZE;
		int Mat2index = (B*BLOCK_SIZE + row)*Bcols + BLOCK_SIZE*blockcol + col;

		if (Mat1index < Arows*Acols)		
			subAshared[row*BLOCK_SIZE + col] = Mat1[Mat1index];     
 		else
			subAshared[row*BLOCK_SIZE + col] = 0;

		if (Mat2index < Acols*Bcols)     
			subBshared[row*BLOCK_SIZE + col] = Mat2[Mat2index]; 
		else
			subBshared[row*BLOCK_SIZE + col] = 0;

		__syncthreads();
					
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			if ((row*BLOCK_SIZE + j < BLOCK_SIZE*BLOCK_SIZE) && (j*BLOCK_SIZE + col < BLOCK_SIZE*BLOCK_SIZE))
			{
				Cvalue += subAshared[row*BLOCK_SIZE + j]*subBshared[j*BLOCK_SIZE + col];
			}
		}

		__syncthreads();
			
	}
	if ((row < Arows) && (col < Bcols))
    {
		int finalmatrow = blockrow*BLOCK_SIZE + row;
		int finalmatcol = blockcol*BLOCK_SIZE + col;
		OutMat[finalmatrow*Bcols +  finalmatcol] = Cvalue;
	}
}




