#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void addKernel(int a, int b, int *c)
{
    *c = (a+b)*(a+b);
}

__global__ void setVectorKernel(int *v, int g)
{
	v[threadIdx.x + (blockDim.x * blockIdx.x)] = g;
}

__global__ void addVectorKernel(int* a, int* b, int* out, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<N)
		out[i] = a[i] + b[i];
}

__global__ void dwtKernel(float* input, float* output)
{
	const int VECTOR_SIZE = 32;
	const int VECTOR_HALF_SIZE = 16;

	__shared__ float res[VECTOR_SIZE];
	__shared__ float res2[VECTOR_SIZE];

	// scaled and coefficients for Daubechies 4 wavelet
	const int WaveletLength = 4;
	float Coefficients[WaveletLength];
    Coefficients[0] = 0.4829629131;
    Coefficients[1] = 0.8365163037;
    Coefficients[2] = 0.2241438680;
    Coefficients[3] = -0.1294095226;
    float Scales[WaveletLength];
    Scales[0] = Coefficients[3];
    Scales[1] = -Coefficients[2];
    Scales[2] = Coefficients[1];
    Scales[3] = -Coefficients[0];

	int dx= threadIdx.x;

	int k = 0;
	for (int i = 0; i < WaveletLength; i++)
	{
		k = (dx*2)+i;
		if(k>=VECTOR_SIZE){k -= VECTOR_SIZE;}

		// set to zero for smoothing purposes, otherwise use commented formula
		res[dx] += 0; // input[k] * Scales[i];
		res[dx + VECTOR_HALF_SIZE] += input[k]*Coefficients[i];
	}

	// wait for DWT
	__syncthreads();

	for (int i = 0; i < WaveletLength; i++)
	{
		k = (dx*2)+i;
		if(k>=VECTOR_SIZE) {k-=VECTOR_SIZE;}
		res2[k] += (res[dx] *Scales[i] + res[dx + VECTOR_HALF_SIZE] * Coefficients[i]);
	}

	// wait for inverse transform
	__syncthreads();

	output[dx] = res2[dx];
	output[dx + VECTOR_HALF_SIZE] = res2[dx+VECTOR_HALF_SIZE];
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 1; i <= 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i-1]);
    for (int i = 1; i <= 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i-1]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

int main()
{
	// Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
 
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
 
    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);
    return 0;
}

