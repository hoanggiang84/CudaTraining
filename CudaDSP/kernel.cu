#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "GpuTimer.h"

typedef float2 Complex;
#define SIGNAL_SIZE  1000000
#define SIGNAL_NUM   1
#define MAX_THREADS_PER_BLOCK 1024

__global__ void hilbert_intermediate_transform(cufftComplex* d_fft)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= SIGNAL_SIZE)
		return;

	if(idx <= SIGNAL_SIZE/2)
	{
		d_fft[idx].x = d_fft[idx].x * 2;
		d_fft[idx].y = d_fft[idx].y * 2;
	}
	else
	{
		d_fft[idx].x = 0;
		d_fft[idx].y = 0;
	}
} 

__global__ void hilbert_final_transform(cufftComplex* d_fft, cufftComplex* d_original)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < SIGNAL_SIZE)
	{
		d_fft[idx].x = d_original[idx].x ;
		d_fft[idx].y = d_fft[idx].y / SIGNAL_SIZE;
	}
} 

void runDemo()
{
	GpuTimer timer;
	//float samples[10] = {-0.1, 0.2,0.3,-0.2,-0.1, 0.2,0.3,-0.2,-0.4, 0.01};
	int num_blocks = SIGNAL_SIZE / MAX_THREADS_PER_BLOCK + 1;
	int mem_size = sizeof(Complex) * SIGNAL_SIZE;

    // Allocate host memory for the signal
    Complex* h_signal = (Complex*)malloc(mem_size);

	printf("Initialize sample signal...\n");
    // Initalize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i].x = rand()/float(RAND_MAX);
		//h_signal[i].x = samples[i];
        h_signal[i].y = 0;
		//printf("(%2.2f\t %2.2f)\n", h_signal[i].x, h_signal[i].y);
    }

	// Allocate device memory for signal
	Complex* d_signal;
	Complex* d_original;
    cudaMalloc((void**)&d_signal, mem_size);
	cudaMalloc((void**)&d_original, mem_size);
	cudaHostRegister(h_signal, mem_size, cudaHostRegisterPortable);

    cufftHandle plan;
    cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1);

	printf("Hilbert transform is starting...\n");
	timer.Start();

	cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_original, h_signal, mem_size, cudaMemcpyHostToDevice);

    // FFT
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

	hilbert_intermediate_transform<<<num_blocks, MAX_THREADS_PER_BLOCK>>>((cufftComplex *)d_signal);

	// IFFT
	cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);
		
	hilbert_final_transform<<<num_blocks,MAX_THREADS_PER_BLOCK>>>((cufftComplex *)d_signal, (cufftComplex *)d_original);

	//cudaDeviceSynchronize();

	cudaMemcpy(h_signal, d_signal, mem_size, cudaMemcpyDeviceToHost);

	timer.Stop();
    
	float ms = timer.Elapsed();
	float magnitude;
	//for (unsigned int i = 0; i < 10; ++i) {
	//	magnitude = sqrtf(powf(h_signal[i].x,2.0f) + powf(h_signal[i].y,2.0f));
	//	printf("(%2.2f\t %2.2f),\t %2.2f\n", h_signal[i].x, h_signal[i].y, magnitude);
	//	//printf("(%f, %f)\n", h_signal[i].x, h_signal[i].y);
	//}

	printf("Fast Hilbert Transform Done! Time Elapsed %f ms\n", ms);
	// Clean up 
	cudaHostUnregister(h_signal);
    cufftDestroy(plan);
    free(h_signal);
    cudaFree(d_signal);
}

int main()
{
	runDemo();
    return 0;
}

