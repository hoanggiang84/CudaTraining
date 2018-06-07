#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "GpuTimer.h"
#include <stdio.h>

#define NUM_THREADS 1000000
#define ARRAY_SIZE 10

#define BLOCK_WIDTH 1000

__global__ void increment_naive(int *g)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	 i = i % ARRAY_SIZE;
	 g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	 i = i % ARRAY_SIZE;
	 atomicAdd(&g[i],1);
}

int main()
{
	GpuTimer timer;
	printf("%d total threads in %d blocks writing into %d array elements\n",
		NUM_THREADS, NUM_THREADS/BLOCK_WIDTH, ARRAY_SIZE);

	// declare and allocate host memory
	int h_arr[ARRAY_SIZE];
	const int ARRAY_BYTES = ARRAY_SIZE*sizeof(int);

	// declare, allocate, and zero out GPU memory
	int *d_arr;
	cudaMalloc((void**)&d_arr, ARRAY_BYTES);
	cudaMemset(d_arr, 0, ARRAY_BYTES);

	// launch the kernal
	timer.Start();
	increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_arr);
	timer.Stop();

	// copy back the array of sums from GPU and print
	cudaMemcpy(h_arr,d_arr, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < ARRAY_SIZE; i++)
		printf("%d \n", h_arr[i]);
	
	printf("Time elapsed = %g ms\n", timer.Elapsed());

	// free GPU memory allocation and exit
	cudaFree(d_arr);

    return 0;
}
