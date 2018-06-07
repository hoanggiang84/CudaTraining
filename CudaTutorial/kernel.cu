#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
    printf("Hello world! I'm a thread in block %d\n",blockIdx.x);
}

__global__ void use_local_memory_GPU(float in)
{
	float f;
	f = in;
}

__global__ void use_global_memory_GPU(float *arr)
{
	arr[threadIdx.x] = 2.0f * float(threadIdx.x);
}

__global__ void use_shared_memory_GPU(float *arr)
{
	int i, index = threadIdx.x;
	float average, sum = 0.0f;

	__shared__ float sh_arr[128];

	sh_arr[index] = arr[index];

	__syncthreads();

	for (int i = 0; i < index; i++)
	{
		sum += sh_arr[i];
	}
	average = sum/(index +1.0f);

	if(arr[index] > average){arr[index] = average;}

	sh_arr[index] = 3.14;
}

int main()
{
	// launch the kernel
	//hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
	// force the printf()s to flush
	//cudaDeviceSynchronize();

	//use_local_memory_GPU<<<1,128>>>(2.0f);

	float h_arr[128];
	for (int i = 0; i < 128; i++)
		h_arr[i] = float(i);

	float *d_arr;

	// allocate global memory on the device, place result in "d_arr"
	int array_bites = 128*sizeof(float);
	cudaMalloc((void**)&d_arr, array_bites);

	// transfer the array to the GPU
	cudaMemcpy(d_arr, h_arr, array_bites, cudaMemcpyHostToDevice);

	// launch the kernal
	use_global_memory_GPU<<<1, 128>>>(d_arr);

	// copy back the result to the CPU
	cudaMemcpy(h_arr, d_arr, array_bites, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 128; i++)
		printf("%f\n", h_arr[i]);

	use_shared_memory_GPU<<<1,128>>>(d_arr);
	cudaMemcpy(h_arr, d_arr, array_bites, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 128; i++)
		printf("%f\n", h_arr[i]);

	cudaFree(d_arr);
	printf("That's all!\n");
    return 0;
}

