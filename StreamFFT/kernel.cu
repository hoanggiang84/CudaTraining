#include <string.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GpuTimer.h"

#define NUM_STREAMS 3

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/********/
/* MAIN */
/********/
int main()
{
    const int N = 500000;

    // --- Host input data initialization
    float2 *h_in1 = new float2[N];
    float2 *h_in2 = new float2[N];
    float2 *h_in3 = new float2[N];
    for (int i = 0; i < N; i++) {
        h_in1[i].x = 1.f;
        h_in1[i].y = 0.f;
        h_in2[i].x = 1.f;
        h_in2[i].y = 0.f;
        h_in3[i].x = 1.f;
        h_in3[i].y = 0.f;
    }

    // --- Host output data initialization
    float2 *h_out1 = new float2[N];
    float2 *h_out2 = new float2[N];
    float2 *h_out3 = new float2[N];
    for (int i = 0; i < N; i++) {
        h_out1[i].x = 0.f;
        h_out1[i].y = 0.f;
        h_out2[i].x = 0.f;
        h_out2[i].y = 0.f;
        h_out3[i].x = 0.f;
        h_out3[i].y = 0.f;
    }

    // --- Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
    gpuErrchk(cudaHostRegister(h_in1, N*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_in2, N*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_in3, N*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_out1, N*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_out2, N*sizeof(float2), cudaHostRegisterPortable));
    gpuErrchk(cudaHostRegister(h_out3, N*sizeof(float2), cudaHostRegisterPortable));

    // --- Device input data allocation
    float2 *d_in1;          gpuErrchk(cudaMalloc((void**)&d_in1, N*sizeof(float2)));
    float2 *d_in2;          gpuErrchk(cudaMalloc((void**)&d_in2, N*sizeof(float2)));
    float2 *d_in3;          gpuErrchk(cudaMalloc((void**)&d_in3, N*sizeof(float2)));
    float2 *d_out1;         gpuErrchk(cudaMalloc((void**)&d_out1, N*sizeof(float2)));
    float2 *d_out2;         gpuErrchk(cudaMalloc((void**)&d_out2, N*sizeof(float2)));
    float2 *d_out3;         gpuErrchk(cudaMalloc((void**)&d_out3, N*sizeof(float2)));

    // --- Creates CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

    // --- Creates cuFFT plans and sets them in streams
    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cufftPlan1d(&plans[i], N, CUFFT_C2C, 1);
        cufftSetStream(plans[i], streams[i]);
    }

	GpuTimer timer;
	timer.Start();
    // --- Async memcopyes and computations
    gpuErrchk(cudaMemcpyAsync(d_in1, h_in1, N*sizeof(float2), cudaMemcpyHostToDevice, streams[0]));
    gpuErrchk(cudaMemcpyAsync(d_in2, h_in2, N*sizeof(float2), cudaMemcpyHostToDevice, streams[1]));
    gpuErrchk(cudaMemcpyAsync(d_in3, h_in3, N*sizeof(float2), cudaMemcpyHostToDevice, streams[2]));
    cufftExecC2C(plans[0], (cufftComplex*)d_in1, (cufftComplex*)d_out1, CUFFT_FORWARD);
    cufftExecC2C(plans[1], (cufftComplex*)d_in2, (cufftComplex*)d_out2, CUFFT_FORWARD);
	cufftExecC2C(plans[2], (cufftComplex*)d_in3, (cufftComplex*)d_out3, CUFFT_FORWARD);
	cufftExecC2C(plans[0], (cufftComplex*)d_out1, (cufftComplex*)d_out1, CUFFT_INVERSE);
    cufftExecC2C(plans[1], (cufftComplex*)d_out2, (cufftComplex*)d_out2, CUFFT_INVERSE);
    cufftExecC2C(plans[2], (cufftComplex*)d_out3, (cufftComplex*)d_out3, CUFFT_INVERSE);
    gpuErrchk(cudaMemcpyAsync(h_out1, d_out1, N*sizeof(float2), cudaMemcpyDeviceToHost, streams[0]));
    gpuErrchk(cudaMemcpyAsync(h_out2, d_out2, N*sizeof(float2), cudaMemcpyDeviceToHost, streams[1]));
    gpuErrchk(cudaMemcpyAsync(h_out3, d_out3, N*sizeof(float2), cudaMemcpyDeviceToHost, streams[2]));

    for(int i = 0; i < NUM_STREAMS; i++)
        gpuErrchk(cudaStreamSynchronize(streams[i]));

	timer.Stop();
	float ms = timer.Elapsed();
	
	printf("Stream Fast Fourier Transform. Time Elapsed: %f ms\n", ms);

    // --- Releases resources
    gpuErrchk(cudaHostUnregister(h_in1));
    gpuErrchk(cudaHostUnregister(h_in2));
    gpuErrchk(cudaHostUnregister(h_in3));
    gpuErrchk(cudaHostUnregister(h_out1));
    gpuErrchk(cudaHostUnregister(h_out2));
    gpuErrchk(cudaHostUnregister(h_out3));
    gpuErrchk(cudaFree(d_in1));
    gpuErrchk(cudaFree(d_in2));
    gpuErrchk(cudaFree(d_in3));
    gpuErrchk(cudaFree(d_out1));
    gpuErrchk(cudaFree(d_out2));
    gpuErrchk(cudaFree(d_out3));

    for(int i = 0; i < NUM_STREAMS; i++) 
		gpuErrchk(cudaStreamDestroy(streams[i]));

    delete[] h_in1;
    delete[] h_in2;
    delete[] h_in3;
    delete[] h_out1;
    delete[] h_out2;
    delete[] h_out3;
	
    //cudaDeviceReset();  
	printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);

    return 0;
}