#include <string.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GpuTimer.h"

typedef float2 Complex;

#define NUM_STREAMS 4
#define NUM_RECORDS 2000
#define RECORD_LENGTH 500

__global__ void hit(cufftComplex* d_matrix, int num_records, int record_length)
{
	int signalIndex = threadIdx.x; 
	if(signalIndex >= record_length)
		return;

	int recordIndex = blockIdx.x;
	int idx = recordIndex*record_length + signalIndex;
	float2 temp = d_matrix[idx];
	if(signalIndex <= record_length/2)
	{
		temp.x = temp.x * 2;
		temp.y = temp.y * 2;
		d_matrix[idx] = temp;
	}
	else if(signalIndex < record_length)
	{
		temp.x = 0;
		temp.y = 0;
		d_matrix[idx] = temp;
	}
}

__global__ void hft(cufftComplex* d_matrix, cufftComplex* d_original, int num_record, int record_length)
{
	int signalIndex = threadIdx.x; 
	if(signalIndex >= record_length)
		return;

	int recordIndex = blockIdx.x;
	int idx = recordIndex*record_length + signalIndex;
	//d_matrix[idx].x = d_original[idx].x ;
	d_matrix[idx].y = d_matrix[idx].y / record_length;
	d_matrix[idx].x = sqrtf(powf(d_original[idx].x,2) + powf(d_matrix[idx].y,2));
} 

__global__ void max_kernel(float * d_out, cufftComplex * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // Compare elements in first half with second half
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
			if(d_in[myId + s].x > d_in[myId].x)
				d_in[myId].x = d_in[myId + s].x;
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId].x;
    }
}

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
	int sampleNum = NUM_RECORDS * RECORD_LENGTH;
	int mem_size = sizeof(Complex) * sampleNum;

	Complex* h_matrix = (Complex*)malloc(mem_size);
    for (int j=0; j<NUM_RECORDS; j++) 
	{
		for (int i=0; i<RECORD_LENGTH; i++) {
            h_matrix[j*RECORD_LENGTH+i].x = rand()/float(RAND_MAX);
			h_matrix[j*RECORD_LENGTH+i].y = 0;
        }
	}

	Complex* d_matrix;
	Complex* d_original;
	float* h_max = (float*)malloc(NUM_RECORDS*sizeof(float));
	float* d_max;

	gpuErrchk(cudaHostRegister(h_matrix, mem_size, cudaHostRegisterPortable));
	gpuErrchk(cudaHostRegister(h_max, NUM_RECORDS*sizeof(float), cudaHostRegisterPortable));
	gpuErrchk(cudaMalloc((void**)&d_matrix, mem_size));
	gpuErrchk(cudaMalloc((void**)&d_original, mem_size));
	gpuErrchk(cudaMalloc((void**)&d_max, NUM_RECORDS*sizeof(float)));

	// --- Creates CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) 
		gpuErrchk(cudaStreamCreate(&streams[i]));

    // --- Creates cuFFT plans and sets them in streams
    // --- Advanced data layout
    //     input[b * idist + x * istride]
    //     output[b * odist + x * ostride]
    //     b = signal number
    //     x = element of the b-th signal

    int rank = 1;                           // --- 1D FFTs
    int n[] = { RECORD_LENGTH };                        // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = RECORD_LENGTH, odist = RECORD_LENGTH;               // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = NUM_RECORDS/NUM_STREAMS;                      // --- Number of batched executions

    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
	int streamSize = sampleNum/NUM_STREAMS;
    for (int i = 0; i < NUM_STREAMS; i++) {
        //cufftPlan1d(&plans[i], streamSize, CUFFT_C2C, 1);
		cufftPlanMany(&plans[i], rank, n, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2C, batch);
        cufftSetStream(plans[i], streams[i]);
    }

	GpuTimer timer;
	timer.Start();

    // --- Async memcopyes and computations
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		int offset = i * streamSize;
		gpuErrchk(cudaMemcpyAsync(&d_matrix[offset], &h_matrix[offset], streamSize*sizeof(float2), cudaMemcpyHostToDevice, streams[i]));
		gpuErrchk(cudaMemcpyAsync(&d_original[offset], &h_matrix[offset], streamSize*sizeof(float2), cudaMemcpyHostToDevice, streams[i]));
	}
    
	int blockSize = RECORD_LENGTH;
	int gridSize = batch;
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		int offset = i * streamSize;
		cufftExecC2C(plans[i], (cufftComplex*)(&d_matrix[offset]), (cufftComplex*)(&d_matrix[offset]), CUFFT_FORWARD);
		hit<<<gridSize, blockSize, 0, streams[i]>>>((cufftComplex*)(&d_matrix[offset]), streamSize, RECORD_LENGTH);
		cufftExecC2C(plans[i], (cufftComplex*)(&d_matrix[offset]), (cufftComplex*)(&d_matrix[offset]), CUFFT_INVERSE);
		hft<<<gridSize, blockSize, 0, streams[i]>>>((cufftComplex*)(&d_matrix[offset]), (cufftComplex*)(&d_original[offset]), streamSize, RECORD_LENGTH);

		int offset2 = i * batch;
		max_kernel<<<gridSize, blockSize, 0, streams[i]>>>((float*)(&d_max[offset2]), (cufftComplex*)(&d_matrix[offset]));
	}

	for (int i = 0; i < NUM_STREAMS; i++)
	{
		int offset = i * streamSize;
		gpuErrchk(cudaMemcpyAsync(&h_matrix[offset], &d_matrix[offset], streamSize*sizeof(float2), cudaMemcpyDeviceToHost, streams[i]));

		int offset2 = i * batch;
		gpuErrchk(cudaMemcpyAsync(&h_max[offset2], &d_max[offset2], batch*sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
	}

    for(int i = 0; i < NUM_STREAMS; i++)
        gpuErrchk(cudaStreamSynchronize(streams[i]));

	timer.Stop();
	float ms = timer.Elapsed();
   
	cudaFree(d_matrix);
	cudaFree(d_original);
	cudaFree(d_max);
	cudaHostUnregister(h_matrix);
	cudaHostUnregister(h_max);
	free(h_matrix);
	free(h_max);

	for(int i = 0; i < NUM_STREAMS; i++) 
		gpuErrchk(cudaStreamDestroy(streams[i]));

	printf("Stream Fast Fourier Transform. Time Elapsed: %fms", ms);
	printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);

    return 0;
}