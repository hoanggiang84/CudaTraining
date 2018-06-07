//#include <thrust/device_vector.h>
#include <string.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GpuTimer.h"

typedef float2 Complex;

#define NUM_RECORDS 2000
#define RECORD_LENGTH 500
#define MAX_THREADS_PER_BLOCK 512

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
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
	int blockSize = RECORD_LENGTH;
	int gridSize = NUM_RECORDS;
	//float samples[10] = {-0.1, 0.2,0.3,-0.2,-0.1, 0.2,0.3,-0.2,-0.4, 0.01};

	int mem_size = sizeof(Complex)*NUM_RECORDS*RECORD_LENGTH;
	Complex* h_matrix = (Complex*)malloc(mem_size);
    for (int j=0; j<NUM_RECORDS; j++) 
	{
		for (int i=0; i<RECORD_LENGTH; i++) {
            float2 temp;
			temp.x = rand()/float(RAND_MAX);
			//temp.x = samples[i];
            temp.y = 0.f;
            h_matrix[j*RECORD_LENGTH+i] = temp;
            if(j==0 && RECORD_LENGTH < 100)
				printf("(%2.2f\t %2.2f) \n", temp.x, temp.y); 
        }
		//printf("\n");
	}

    // --- Advanced data layout
    //     input[b * idist + x * istride]
    //     output[b * odist + x * ostride]
    //     b = signal number
    //     x = element of the b-th signal

    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { RECORD_LENGTH };                        // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = RECORD_LENGTH, odist = RECORD_LENGTH;               // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = NUM_RECORDS;                      // --- Number of batched executions
    cufftPlanMany(&handle, rank, n, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2C, batch);

	GpuTimer timer;
	Complex* h_pinned;
	Complex* d_matrix;
	Complex* d_original;
	float* h_max = (float*)malloc(NUM_RECORDS*sizeof(float));
	float* d_max;

	cudaMallocHost((void**)&h_pinned, mem_size);
    cudaMalloc((void**)&d_matrix, mem_size);
	cudaMalloc((void**)&d_original, mem_size);
	cudaMalloc((void**)&d_max, NUM_RECORDS*sizeof(float));

	timer.Start();

	// Pinned Host Memory
	memcpy(h_pinned, h_matrix, mem_size);
	cudaMemcpy(d_matrix, h_pinned, mem_size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_original, h_pinned, mem_size, cudaMemcpyHostToDevice);
	//- Pinned Host Memory

	//// Pageable Host Memory
	//cudaMemcpy(d_matrix, h_matrix, mem_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_original, h_matrix, mem_size, cudaMemcpyHostToDevice);
	////- Pageable Host Memory

	cufftExecC2C(handle, (cufftComplex*)(d_matrix), (cufftComplex*)(d_matrix), CUFFT_FORWARD);
	
	hit<<<gridSize, blockSize>>>((cufftComplex*)(d_matrix), NUM_RECORDS, RECORD_LENGTH);

	cufftExecC2C(handle, (cufftComplex*)(d_matrix), (cufftComplex*)(d_matrix), CUFFT_INVERSE);

	hft<<<gridSize, blockSize>>>((cufftComplex*)(d_matrix), (cufftComplex*)(d_original), NUM_RECORDS, RECORD_LENGTH);

	max_kernel<<<gridSize, blockSize>>>(d_max, (cufftComplex*)(d_matrix));

	cudaMemcpy(h_max, d_max, NUM_RECORDS*sizeof(float), cudaMemcpyDeviceToHost);

	// Pinned Host Memory
	cudaMemcpy(h_pinned, d_matrix, mem_size, cudaMemcpyDeviceToHost);
	//- Pinned Host Memory

	//// Pageable Host Memory
	//cudaMemcpy(h_matrix, d_matrix, mem_size, cudaMemcpyDeviceToHost);
	////- Pageable Host Memory

	timer.Stop();
	float ms = timer.Elapsed();

	printf("Matrix Transform Done! Time Elapsed: %f ms\n", ms);
	float magnitude;
	float max =0;
	for (int j=0; j<NUM_RECORDS; j++) 
	{
		for (int i=0; i<RECORD_LENGTH; i++) 
		{ 
			float2 temp = h_pinned[j*RECORD_LENGTH+i];
			if(j==0)// && RECORD_LENGTH < 100)
			{
				//magnitude = sqrtf(powf(temp.x,2.0f) + powf(temp.y,2.0f));
				//printf("(%2.2f\t %2.2f), %2.2f\n", temp.x, temp.y);
				if(temp.x > max)
					max = temp.x;
			}
		}
		if(j==0)
			printf("%f  %f\n", h_max[j], max);
	}

	cufftDestroy(handle);
	free(h_matrix);
	cudaFree(d_matrix);
	cudaFree(d_original);
	cudaFreeHost(h_pinned);
}