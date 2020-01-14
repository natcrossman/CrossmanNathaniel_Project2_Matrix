
/**
*@copyright     All rights are reserved, this code/project is not Open Source or Free
*@author        Nathaniel Crossman (U00828694)
*@email		 crossman.4@wright.edu
*
*@Professor     Meilin Liu
*@Course_Number CS 4370/6370-90
*@Date			 Thursday, Oct 21, 2019
*
*@project_name:
*				Task 1: Basic Matrix Addition
*				Task 2: Basic Matrix Multiplication
*
*
*/
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
// CUDA runtime
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <string.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>



/**
* Matrix multiplication on the device with tiled matrix : C = A * B
* Width is same for both dim
*/
template <int TILE_WIDTH> __global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int width){

	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;  
	int tx = threadIdx.x;

	int by = blockIdx.y;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	for (int m = 0; m < width / TILE_WIDTH; ++m){
		//Load the matrices from device memory to shared memory
		//Each thread loads one element of each matrix
		ds_M[ty][tx] = d_M[Row*width + m * TILE_WIDTH + tx];
		ds_N[ty][tx] = d_N[Col + (m*TILE_WIDTH + ty)*width];
		
		// Synchronize to make sure the matrices are loaded
		__syncthreads();


		// Multiply the two matrices together
		for (int k = 0; k < TILE_WIDTH; ++k) {
			//Each thread computes only one element of the block TILE_WIDTH
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		}
		// We need to make sure that the preceding computation is done before loading two new
		// TILE_WIDTH of d_M and d_N in the next iteration
		__syncthreads();
	
	}

	d_P[Row*width + Col] = Pvalue;

	
}


//below is all prototypes
void matrixMulOnHost(float* M, float* N, float* P, int width);
void multiplicationMatrixMain();

//for testing floats
int verify(float *matrix_A, float *matrix_B, float *matrix_C, int width);
int verifyNew(float *cpu_matrix_C, float *gpu_matrix_C, int width);
int isRelativeOrAbsoluteEqualFloat(float A, float B, float maxRelativeError, float maxAbsoluteError);

//Helper f
int menuShow();
void mainSwitch(int option);
void cls();
int debugOn();
void getBlockSize(int &tileWidth);
void getWidth(int &width);

void printf_matrix(float *matrix_A, int width);

//Above is all prototypes

int main()
{	
	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line
	//int dev = findCudaDevice(argc, (const char **)argv);
	while (true) {
		mainSwitch(menuShow());
		printf("\n");
	}
	return 0;
}
int menuShow() {
	int hold;
	do {
		printf("1. Tiled Matrix  \n");
		printf("2. Quit\n");
		printf("---------------------------------------\n");
		printf("Enter Choice: ");
		scanf("%d", &hold);

		if (hold < 1 || hold > 2) {
			cls();
		}
	} while (hold < 1 || hold > 2);
	return hold;
}
void cls() {
	for (int i = 0; i < 30; ++i)
			printf("\n");
	system("@cls||clear");
}
void mainSwitch(int option) {
	switch (option) {
	case 1:
		multiplicationMatrixMain();
		break;
	case 2:
		exit(0);
		break;
	}
}
void getWidth(int &width) {
	printf("Please specify your square matrix dimension\n");
	printf("For example, you could enter 8 and the matrix dimension 8*8\n");
	printf("Enter Square Matrix size:");
	scanf("%d", &width);
	cls();
}
void getBlockSize(int &tileWidth) {
	printf("Please specify your Thread block / (tile-width) \n");
	printf("For example, you could enter 4 and the block size would be 4 * 4 \n");
	printf("Enter Block Size:");
	scanf("%d", &tileWidth);
	cls();
	
}
void printf_matrix(float *matrix_A, int width) {
	int i, j, index;
	for (i = 0; i < width; ++i)
	{
		for (j = 0; j < width; ++j) {
			index = i * width + j;
			printf("%f \t", matrix_A[index]);
		}
		printf("\n");
	}
	printf("\n");
}
int verify(float *matrix_A, float *matrix_B, float *matrix_C, int width) {
	const float relativeTolerance = 1e-6; // 1e-6 = 0.000001
	for (int row = 0; row < width; ++row){
		for (int col = 0; col < width; ++col){
			float sum = 0;
			for (int k = 0; k < width; ++k){
				sum += matrix_A[row*width + k] * matrix_B[k*width + col];
			}

			float relativeError = (sum - matrix_C[row*width + col]) / sum;

			if (relativeError > relativeTolerance || relativeError < -relativeTolerance){
				printf("TEST FAILED \n\n");
				printf("RelativeError is: %f \n", relativeError);
				printf("RelativeTolerance is: %f \n", relativeTolerance);
				return 0;
			}
		}
	}
	printf("TEST PASSED\n\n");

	return 1;
}

/*
Why this function is better!
We do not need to loop over all items in both A and B matrix! And the float comparison is more precise.
Prof. Told me this was fine to do for this project..

When do an approximate comparison of floats, one want it to be relative to the magnitude of the numbers. In other words, you’re basically interested in the fact that they agree (or don’t agree) to some particular number of significant digits. For example, you might decide that they need to agree to at least 3 decimal digits to be considered equal.
f(){
if (fabs(result - expectedResult) < 0.00001)
}
Even now our function isn’t perfect. In general this function will behave poorly for numbers around zero. The positive number closest to zero and the negative number closest to zero are extremely close to each other, yet this function will correctly calculate that they have a huge relative error of 2.0. If you want to count numbers near zero but of opposite sign as being equal then you need to add a maxAbsoluteError check also. The function would then return true if either the absoluteError or the relativeError were smaller than the maximums passed in. A typical value for this backup maxAbsoluteError would be very small – FLT_MAX or less, depending on whether the platform supports subnormals.

*/
int verifyNew(float *cpu_matrix_C, float *gpu_matrix_C, int width) {
	const float relativeTolerance = 1e-6; // 1e-6 = 0.000001
	const float maxAbsoluteError = 1E-37;
	int index = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			//index = i + j * width;
			index = i * width + j;
			if (isRelativeOrAbsoluteEqualFloat(cpu_matrix_C[index], gpu_matrix_C[index], relativeTolerance, maxAbsoluteError)) {
				printf("TEST PASSED\n\n");
				return 1;
			}
			else {
				printf("TEST FAILED \n\n");
				return 0;
			}
		}
	}
	printf("TEST FAILED \n\n");
	return 0;

}


int isRelativeOrAbsoluteEqualFloat(float A, float B, float maxRelativeError, float maxAbsoluteError){
	if (fabsf(A) == fabsf(0.0f) || fabsf(B) == fabsf(0.0f)) {
		//very small flt_max
		return 0;
	}
	if (fabsf(A - B) < maxAbsoluteError) {
		//very small flt_max
		return 1;
	}
	float relativeError = 0.0f;

	if (fabsf(B) > fabsf(A)) {
		relativeError = fabsf((A - B) / B);
	}else {
		relativeError = fabsf((A - B) / A);
	}
	if (relativeError <= maxRelativeError) {
		return 1;
	}
	return 0;
}

void matrixMulOnHost(float* M, float* N, float* P, int width){
	for (int i = 0; i<width; ++i){
		for (int j = 0; j < width; ++j){
			float sum = 0;
			for (int k = 0; k < width; ++k){
				float a = M[i * width + k];
				float b = N[k * width + j];
				sum += a * b;
			}
			P[i * width + j] = sum;
		}
	}
}

int debugOn() {
	int hold;
	do {
		printf("\nRun in debug mode?\n");
		printf("Debug mode prints out alot of helpful info,\nbut it can takes a long time with big matrixes\n");
		printf("Enter 1 for Yes and 0 for No:");
		scanf("%d", &hold);
		if (hold < 0 || hold > 1) {
			cls();
		}
	} while (hold < 0 || hold > 1);
	cls();
	return hold;
}


void multiplicationMatrixMain() {
	int width = 0, tileWidth = 0;
	float secTotal = 0.0f;
	float *h_matrix_A, *h_matrix_B, *h_matrix_C;

	float *d_matrix_A, *d_matrix_B, *d_matrix_C;

	float * h_matrix_final_gpu;

	int booleanValue = debugOn();
	getWidth(width);
	getBlockSize(tileWidth);
	printf("Matrix Size: %d * %d \nSize of Thread block: %d * %d", width, width, tileWidth, tileWidth);
	printf("\n\n");
	cudaError_t cudaError;

	//The size of all matrixes
	size_t dsize = (width * width) * sizeof(float);

	//Allocate memory for matrices on host
	h_matrix_A = (float*)malloc(dsize);
	h_matrix_B = (float*)malloc(dsize);
	h_matrix_C = (float*)malloc(dsize);
	h_matrix_final_gpu = (float*)malloc(dsize);

	memset(h_matrix_A, 0, dsize);
	memset(h_matrix_B, 0, dsize);
	memset(h_matrix_C, 0, dsize);
	memset(h_matrix_final_gpu, 0, dsize);

	if (h_matrix_final_gpu == NULL || h_matrix_C == NULL || h_matrix_B == NULL || h_matrix_A == NULL ) {
		printf("Failed to allocate host matrix C!\n");
	}

	int i = 0, j = 0, index = 0;;
	int init = 1325;
	for (i = 0; i<width; ++i) {
		for (j = 0; j<width; ++j) {
			index = i * width + j;
			init = 3125 * init % 65536;
			h_matrix_A[index] = (init - 32768.0f) / 16384.0f;
			init = 3125 * init % 65536;
			h_matrix_B[index] = (init - 32768.0f) / 16384.0f;
		}
	}


	//Allocate memory for device Matrix
	cudaError = cudaMalloc((void **)(&d_matrix_A), dsize);
	if (cudaError != cudaSuccess)
	{
		printf("  cudaMalloc %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}
	
	cudaError = cudaMalloc((void **)(&d_matrix_B), dsize);

	if (cudaError != cudaSuccess)
	{
		printf(" cudaMalloc %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}
	cudaError = cudaMalloc((void **)(&d_matrix_C), dsize);

	if (cudaError != cudaSuccess)
	{
		printf(" cudaMalloc %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	printf("multiplying CPU....\n");
	//matrix mul on host 
	clock_t startTime, endTime;
	double cpu_time =0.0f;

	startTime = clock();
	
	matrixMulOnHost(h_matrix_A, h_matrix_B, h_matrix_C, width);

	endTime = clock();
	cpu_time = ((double)(endTime - startTime))* 1000.0 / CLOCKS_PER_SEC;
	//-----------------------------------------------------------

	if (booleanValue) {
		printf_matrix(h_matrix_A, width);
		printf_matrix(h_matrix_B, width);
		printf("\nThe results of CPU Multiplication\n");
		printf_matrix(h_matrix_C, width);
	}

	//copy the Matrices from Host to Device
	cudaError = cudaMemcpy(d_matrix_A, h_matrix_A, dsize, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf(" cudaMemcpy 318 %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	cudaError = cudaMemcpy(d_matrix_B, h_matrix_B, dsize, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
	{
		printf(" cudaMemcpy 324 %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}
	cudaDeviceSynchronize();

	dim3 dimBlock(tileWidth, tileWidth);
	dim3 dimGrid(width / dimBlock.x, width/dimBlock.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaDeviceSynchronize();
	switch (tileWidth) {
		case 4:
			//GPU
			printf("runinng matrixMulKernel<4>\n");
			matrixMulKernel<4> << < dimGrid, dimBlock >> >(d_matrix_A, d_matrix_B, d_matrix_C, width);
			break;
		case 8:
			printf("runinng matrixMulKernel<8>\n");
			matrixMulKernel<8> << < dimGrid, dimBlock >> >(d_matrix_A, d_matrix_B, d_matrix_C, width);
			break;
		case 16:
			printf("runinng matrixMulKernel<16>\n");
			matrixMulKernel<16> << < dimGrid, dimBlock >> >(d_matrix_A, d_matrix_B, d_matrix_C, width);
			break;
		case 32:
			printf("runinng matrixMulKernel<32>\n");
			matrixMulKernel<32> << < dimGrid, dimBlock >> >(d_matrix_A, d_matrix_B, d_matrix_C, width);
			break;
		default:
			printf("Error that value is not available: %d  \n\n", tileWidth);
			exit(0);
		}
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		printf(" matrixMulKernel %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}
		
	cudaDeviceSynchronize();
	// Copy result from device to host
	cudaError  = cudaMemcpy(h_matrix_final_gpu, d_matrix_C, dsize, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess)
	{
		printf(" cudaMemcpy 370 %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	cudaDeviceSynchronize();
	printf("GPU done Multiplying Matrixes\n");
	
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&secTotal, start, stop);
	//Delete
	
	cudaEventDestroy(start);
	cudaDeviceSynchronize();

	cudaEventDestroy(stop);
	cudaDeviceSynchronize();
	
	if (booleanValue) {
		printf("\nThe results of GPU Multiplication\n");
		printf_matrix(h_matrix_final_gpu, width);
	}

	printf("\nVerifying\n");
	

	verifyNew(h_matrix_C, h_matrix_final_gpu, width);

	cudaFree(d_matrix_A);
	cudaDeviceSynchronize();
	cudaFree(d_matrix_B);
	cudaDeviceSynchronize();
	cudaFree(d_matrix_C);

	//Clean up memory
	free(h_matrix_A);
	free(h_matrix_B);
	free(h_matrix_C);
	free(h_matrix_final_gpu);


	printf("Execution Time for GPU: %.5f ms\n", secTotal);
	printf("Execution Time for CPU: %.5f ms\n", cpu_time);
	printf("Speedup : %.5f ms\n", cpu_time/ secTotal);


}

