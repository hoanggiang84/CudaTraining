using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using SAM.Core.DataProcessing;

namespace CudaDemo
{
    class Program
    {
        private const int VECTOR_SIZE = 5120;
        private const int THREADS_PER_BLOCK = 512;

        //private static readonly string kernelPath =
        //    @"C:\Users\GIANG\Documents\Visual Studio 2012\Projects\CudaCSharp\SimpleKernel\x64\Debug\kernel.ptx";

        //private static CudaKernel addVectorWithCuda;

        //static void InitKernels()
        //{
        //    var context = new CudaContext();
        //    var cumodule = context.LoadModule(kernelPath);
        //    addVectorWithCuda = new CudaKernel("_Z15addVectorKernelPiS_S_i", cumodule, context)
        //                            {
        //                                BlockDimensions = THREADS_PER_BLOCK,
        //                                GridDimensions = VECTOR_SIZE/THREADS_PER_BLOCK + 1
        //                            };
        //}

        //private static readonly Func<int[], int[], int, int[]> addVector = (a, b, size) =>
        //                                                       {
        //                                                           CudaDeviceVariable<int> vectorDeviceA = a;
        //                                                           CudaDeviceVariable<int> vectorDeviceB = b;
        //                                                           CudaDeviceVariable<int> vectorDeviceOut = new CudaDeviceVariable<int>(size);
        //                                                           addVectorWithCuda.Run(vectorDeviceA.DevicePointer, vectorDeviceB.DevicePointer, vectorDeviceOut.DevicePointer, size);
        //                                                           int[] output = new int[size];
        //                                                           vectorDeviceOut.CopyToHost(output);
        //                                                           return output;
        //                                                       };

        //private static CudaKernel smoothTimeSeriesWithCuda;
        //static void InitKernels()
        //{
        //    var context = new CudaContext();
        //    var cumodule = context.LoadModule(kernelPath);
        //    smoothTimeSeriesWithCuda = new CudaKernel("_Z9dwtKernelPfS_", cumodule, context)
        //                            {
        //                                BlockDimensions = 16,
        //                            };
        //}

        //private static readonly Func<float[], float[]> smoothVector = a =>
        //                                                         {
        //                                                             int vectorSize = a.Count();
        //                                                             // init parameters
        //                                                             CudaDeviceVariable<float> vectorDeviceA = a;
        //                                                             CudaDeviceVariable<float> vectorDeviceOut = new CudaDeviceVariable<float>(vectorSize);

        //                                                             // run CUDA method
        //                                                             smoothTimeSeriesWithCuda.Run(
        //                                                                 vectorDeviceA.DevicePointer,
        //                                                                 vectorDeviceOut.DevicePointer);

        //                                                             // copy return to host
        //                                                             float[] output = new float[vectorSize];
        //                                                             vectorDeviceOut.CopyToHost(output);
        //                                                             return output;
        //                                                         };

        #region CUDA
        private static readonly string kernelPath =
            @"C:\Users\GIANG\Documents\Visual Studio 2012\Projects\CudaCSharp\CuFFTRow3DMatrix\x64\Debug\kernel.ptx";

        private static readonly string dataPath =
            @"D:\CurrentWork\Data\29e79\e79\29e79_{0}.dat";

        private const int NUM_RECORDS = 1400;
        private const int RECORD_LENGTH = 150;
        private const int MAX_THREADS_PER_BLOCK = 1024;

        private cufftHandle handle;
        private CudaDeviceVariable<float2> d_matrix = new CudaDeviceVariable<float2>(NUM_RECORDS * RECORD_LENGTH);
        private CudaDeviceVariable<float2> d_original = new CudaDeviceVariable<float2>(NUM_RECORDS * RECORD_LENGTH);


        private static CudaKernel hilbertFinalTransformKernel;
        private static CudaKernel hilbertIntermediateTransformKernel;
        static void InitKernels()
        {
            var context = new CudaContext(0);
            var cumodule = context.LoadModule(kernelPath);
            hilbertFinalTransformKernel = new CudaKernel("_Z23hilbert_final_transformP6float2S0_ii", cumodule, context)
            {
                BlockDimensions = MAX_THREADS_PER_BLOCK,
                GridDimensions = NUM_RECORDS / MAX_THREADS_PER_BLOCK + 1,
            };

            hilbertIntermediateTransformKernel = new CudaKernel("_Z30hilbert_intermediate_transformP6float2ii", cumodule, context)
            {
                BlockDimensions = MAX_THREADS_PER_BLOCK,
                GridDimensions = NUM_RECORDS / MAX_THREADS_PER_BLOCK + 1,
            };
        }
        #endregion

        static void Main()
        {
            Action<Action> measure = (body) =>
            {
                var startTime = DateTime.Now;
                body();
                Console.WriteLine("{0} {1}", DateTime.Now - startTime, Thread.CurrentThread.ManagedThreadId);
            };

            //float[] vector = { 924f, 924.25f, 923.75f, 924.25f, 924.75f, 929.5f, 929.25f, 930f, 930f, 932.5f, 933f, 932.75f, 932.5f, 933f, 923f, 933f, 933.5f, 932.5f, 933.25f, 933.5f, 934.25f, 934.25f, 934.75f, 933f, 932f, 932f, 931.25f, 929.75f, 930f, 932.5f, 931f, 932.25f };
        
            InitKernels();
            //int[] vectorA = Enumerable.Range(1, VECTOR_SIZE).ToArray();
            //int[] vectorB = Enumerable.Range(1, VECTOR_SIZE).ToArray();
            //int[] vector = new int[VECTOR_SIZE];
            //Action act = () => vector = addVector(vectorA,vectorB, VECTOR_SIZE);
            //measure(act);

            //float[] smoothedVector = new float[vector.Length];

            //measure(()=>
            //            {
            //                smoothedVector = smoothVector(vector); 
            //            });

            //Console.WriteLine(vector.Min());
            //Console.WriteLine(smoothedVector.Min());

            //foreach (float d in smoothedVector)
            //{
            //    Console.Write("{0}, ", d);
            //}

            var bScanData = SamBScanLoader.LoadFloat(string.Format(dataPath,1), NUM_RECORDS, RECORD_LENGTH);
            var h_matrix = new float2[NUM_RECORDS*RECORD_LENGTH];
            for (int i = 0; i < NUM_RECORDS; i++)
            {
                for (int j = 0; j < RECORD_LENGTH; j++)
                {
                    h_matrix[i*RECORD_LENGTH + j].x = bScanData[i, j];
                    h_matrix[i*RECORD_LENGTH + j].y = 0;
                    if(i==3)
                        Console.WriteLine("({0},{1})", h_matrix[i*RECORD_LENGTH + j].x, h_matrix[i*RECORD_LENGTH + j].y);
                }
            }

            //cufftHandle handle = new cufftHandle();
            
            int rank = 1;                           // --- 1D FFTs
            int[] n = new[] { RECORD_LENGTH };                        // --- Size of the Fourier transform
            int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
            int idist = RECORD_LENGTH, odist = RECORD_LENGTH;               // --- Distance between batches
            int[] inembed = new[]{ 0 };                  // --- Input size with pitch (ignored for 1D transforms)
            int[] onembed = new[] { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
            int batch = NUM_RECORDS;                      // --- Number of batched executions
            
            //CudaFFTPlanMany fftPlan = new CudaFFTPlanMany(handle, rank, n, batch, cufftType.C2C, inembed, istride, idist, onembed, ostride, odist);
            CudaFFTPlanMany fftPlan = new CudaFFTPlanMany(rank, n, batch, cufftType.C2C, inembed, istride, idist, onembed, ostride, odist);

            CudaDeviceVariable<float2> d_matrix = new CudaDeviceVariable<float2>(NUM_RECORDS * RECORD_LENGTH);
            CudaDeviceVariable<float2> d_original = new CudaDeviceVariable<float2>(NUM_RECORDS * RECORD_LENGTH);

            d_matrix.CopyToDevice(h_matrix);
            d_original.CopyToDevice(h_matrix);

            fftPlan.Exec(d_matrix.DevicePointer, d_matrix.DevicePointer, TransformDirection.Forward);
            hilbertIntermediateTransformKernel.Run(d_matrix.DevicePointer, NUM_RECORDS, RECORD_LENGTH);
            fftPlan.Exec(d_matrix.DevicePointer, d_matrix.DevicePointer, TransformDirection.Inverse);
            hilbertFinalTransformKernel.Run(d_matrix.DevicePointer, d_original.DevicePointer, NUM_RECORDS, RECORD_LENGTH);

            d_matrix.CopyToHost(h_matrix);
            
            fftPlan.Dispose();
            d_matrix.Dispose();
            d_original.Dispose();

            for (int j = 0; j < RECORD_LENGTH; j++)
            {
                h_matrix[3 * RECORD_LENGTH + j].x = bScanData[3, j];
                h_matrix[3 * RECORD_LENGTH + j].y = 0;
                Console.WriteLine("({0},{1})", h_matrix[3 * RECORD_LENGTH + j].x, h_matrix[3 * RECORD_LENGTH + j].y);
            }

            Console.ReadKey();
        }
    }
}
