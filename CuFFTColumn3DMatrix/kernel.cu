#include <thrust/device_vector.h>
#include <cufft.h>
#include <stdlib.h>

#define SIGNAL_SIZE  10

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    const int M = 10;
    const int N = 10;
    const int Q = 1;



    thrust::host_vector<float2> h_matrix(M * N * Q);

    for (int k=0; k<Q; k++) 
        for (int j=0; j<N; j++) 
		{
			for (int i=0; i<M; i++) {
                float2 temp;
                temp.x = rand()/float(RAND_MAX); 
                temp.y = 0.0f;
                h_matrix[k*M*N+j*M+i] = temp;
                printf("(%2.2f %2.2f)\t", temp.x, temp.y); 
            }
			printf("\n");
		}

    thrust::device_vector<float2> d_matrix(h_matrix);

    thrust::device_vector<float2> d_matrix_out(M * N * Q);

    // --- Advanced data layout
    //     input[b * idist + x * istride]
    //     output[b * odist + x * ostride]
    //     b = signal number
    //     x = element of the b-th signal

    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { N };                        // --- Size of the Fourier transform
    int istride = M, ostride = M;           // --- Distance between two successive input/output elements
    int idist = 1, odist = 1;               // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = M;                          // --- Number of batched executions
    cufftPlanMany(&handle, rank, n, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2C, batch);

    for (int k=0; k<Q; k++)
        cufftExecC2C(handle, (cufftComplex*)(thrust::raw_pointer_cast(d_matrix.data()) + k * M * N), (cufftComplex*)(thrust::raw_pointer_cast(d_matrix_out.data()) + k * M * N), CUFFT_FORWARD);
    cufftDestroy(handle);

	printf("\n");
    for (int k=0; k<Q; k++) 
        for (int j=0; j<N; j++) 
		{
			for (int i=0; i<M; i++)  
			{ 
                float2 temp = d_matrix_out[k*M*N+j*M+i];
                printf("(%2.2f %2.2f)\t", temp.x, temp.y); 
            }
			printf("\n");
		}
		

}