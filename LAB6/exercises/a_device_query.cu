/*
 * Assignment 6 — Part A: CUDA Device Query
 * Queries comprehensive GPU device properties.
 */
#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("============================================================\n");
        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("============================================================\n\n");

        /* Q1: Architecture and compute capability */
        printf("1. Compute Capability:          %d.%d\n", prop.major, prop.minor);

        /* Q2: Maximum block dimensions */
        printf("2. Max Block Dimensions:        (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("   Max Threads per Block:       %d\n", prop.maxThreadsPerBlock);

        /* Q3: Maximum grid dimensions */
        printf("3. Max Grid Dimensions:         (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        /* Additional device info */
        printf("   Number of SMs:               %d\n", prop.multiProcessorCount);
        printf("   Max Threads per SM:          %d\n", prop.maxThreadsPerMultiProcessor);

        int clockRate = 0;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        printf("   Clock Rate:                  %.0f MHz\n", clockRate / 1e3);

        /* Q6: Shared memory */
        printf("6. Shared Memory per Block:     %zu bytes (%.0f KB)\n",
               prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
        printf("   Shared Memory per SM:        %zu bytes (%.0f KB)\n",
               prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);

        /* Q7: Global memory */
        printf("7. Global Memory:               %zu bytes (%.0f MB)\n",
               prop.totalGlobalMem, prop.totalGlobalMem / (1024.0 * 1024.0));

        /* Q8: Constant memory */
        printf("8. Constant Memory:             %zu bytes (%.0f KB)\n",
               prop.totalConstMem, prop.totalConstMem / 1024.0);

        /* Q9: Warp size */
        printf("9. Warp Size:                   %d\n", prop.warpSize);

        /* Q10: Double precision support */
        printf("10. Double Precision:           %s (Compute >= 1.3)\n",
               (prop.major > 1 || (prop.major == 1 && prop.minor >= 3)) ? "YES" : "NO");

        /* Extra info */
        int memClockRate = 0;
        cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, dev);

        printf("\n--- Additional Properties ---\n");
        printf("   Memory Clock Rate:           %d kHz (%.0f MHz)\n",
               memClockRate, memClockRate / 1e3);
        printf("   Memory Bus Width:            %d bits\n", prop.memoryBusWidth);
        printf("   L2 Cache Size:               %d bytes (%.0f KB)\n",
               prop.l2CacheSize, prop.l2CacheSize / 1024.0);
        printf("   Max Registers per Block:     %d\n", prop.regsPerBlock);
        printf("   Concurrent Kernels:          %s\n", prop.concurrentKernels ? "YES" : "NO");
        printf("   ECC Enabled:                 %s\n", prop.ECCEnabled ? "YES" : "NO");

        /* Q3 calculation: max threads with grid=65535, block=512 */
        printf("\n--- Q3 Calculation ---\n");
        printf("   If max grid dim = 65535 and max block dim = 512:\n");
        printf("   Max threads = 65535 x 512 = %d\n", 65535 * 512);

        /* Theoretical memory bandwidth */
        double bw_GBs = 2.0 * memClockRate * (prop.memoryBusWidth / 8.0) / 1e6;
        printf("\n   Theoretical Memory BW:       %.1f GB/s\n", bw_GBs);
        printf("============================================================\n");
    }

    return 0;
}
