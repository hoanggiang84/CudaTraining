#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SIZE 1024

__global__ void VectoAdd(int *a, int *b, int *c, int n)
{
	int i = threadIdx.x;
	if (i<n)
	{
		c[i] = a[i] + b[i];
	}
}

__global__ void square(float *dout, float* din)
{
	int idx = threadIdx.x;
	float f = din[idx];
	dout[idx] = f*f;
}

int main()
{
    const int ARRAY_SIZE = 64;
	const int ARRAY_BITES = ARRAY_SIZE*sizeof(float);

	// generate the input array on the host
	float hin[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		hin[i] = float(i);
	}
	float hout[ARRAY_SIZE];

	// declare GPU memory pointers
	float* din;
	float* dout;

	// allocalte GPU memory
	cudaMalloc((void**)&din, ARRAY_BITES);
	cudaMalloc((void**)&dout, ARRAY_BITES);

	// transfer the array to the GPU
	cudaMemcpy(din,hin, ARRAY_BITES, cudaMemcpyHostToDevice);

	// launch the kernel
	square<<<1,ARRAY_SIZE>>>(dout,din);

	// copy back the result array to the CPU
	cudaMemcpy(hout,dout, ARRAY_BITES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for(int i =0; i<ARRAY_SIZE;i++)
	{
		printf("%f", hout[i]);
		printf(((i%4) !=3)? "\t":"\n");
	}

	// free GPU memory allocation
	cudaFree(din);
	cudaFree(dout);

    return 0;
}

