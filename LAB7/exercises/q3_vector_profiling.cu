/*
 * Assignment 7 — Problem 3: Vector Addition with Profiling
 * 1.1 Static global variables (no cudaMalloc)
 * 1.2 Kernel timing with CUDA events
 * 1.3 Theoretical memory bandwidth calculation
 * 1.4 Measured bandwidth of vectorAdd kernel
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define N (1 << 20)   /* 1M elements */
#define THREADS 256

/* 1.1: Statically defined global device variables */
__device__ float d_A_static[N];
__device__ float d_B_static[N];
__device__ float d_C_static[N];

/* Vector addition kernel (used for both static and dynamic versions) */
__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

/* Kernel for static global memory */
__global__ void vectorAddStatic(int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_C_static[i] = d_A_static[i] + d_B_static[i];
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== Assignment 7 — Problem 3: Vector Addition Profiling ===\n");
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("N = %d elements\n\n", N);

    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    int threads = THREADS;
    int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_ms;

    /* ============================================================
     * 1.1: Static global variables (no cudaMalloc)
     * Use cudaMemcpyToSymbol/FromSymbol instead of cudaMemcpy
     * ============================================================ */
    printf("--- 1.1: Static Global Variables ---\n");

    CUDA_CHECK(cudaMemcpyToSymbol(d_A_static, h_A, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_B_static, h_B, bytes));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAddStatic<<<blocks, threads>>>(N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpyFromSymbol(h_C, d_C_static, bytes));

    /* Verify */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) { pass = 0; break; }
    }
    printf("  Static kernel time: %.4f ms  %s\n\n", elapsed_ms, pass ? "[PASS]" : "[FAIL]");

    /* ============================================================
     * 1.2: Timing with CUDA Events (dynamic allocation version)
     * ============================================================ */
    printf("--- 1.2: Kernel Timing (Dynamic) ---\n");

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Warmup */
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed run */
    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    printf("  Dynamic kernel time: %.4f ms\n\n", elapsed_ms);

    /* ============================================================
     * 1.3: Theoretical Memory Bandwidth
     * theoreticalBW = memoryClockRate * memoryBusWidth * 2 (DDR)
     * ============================================================ */
    printf("--- 1.3: Theoretical Memory Bandwidth ---\n");

    int memClockRate = 0;
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, 0);
    int busWidth = prop.memoryBusWidth;

    /* memClockRate is in kHz, busWidth in bits */
    /* BW (kb/s) = memClockRate(kHz) * busWidth(bits) * 2 (DDR) */
    /* Convert: kHz -> Hz (*1e3), bits -> bytes (/8), then /1e9 for GB/s */
    double theoreticalBW = 2.0 * memClockRate * 1e3 * (busWidth / 8.0) / 1e9;

    printf("  Memory Clock Rate:     %d kHz (%.0f MHz)\n", memClockRate, memClockRate / 1e3);
    printf("  Memory Bus Width:      %d bits\n", busWidth);
    printf("  Theoretical BW (DDR):  %.1f GB/s\n\n", theoreticalBW);

    /* ============================================================
     * 1.4: Measured Bandwidth
     * measuredBW = (RBytes + WBytes) / time
     * vectorAdd reads 2 arrays, writes 1 -> 3*N*sizeof(float) bytes
     * ============================================================ */
    printf("--- 1.4: Measured Memory Bandwidth ---\n");

    long long readBytes  = 2LL * N * sizeof(float);  /* read A and B */
    long long writeBytes = 1LL * N * sizeof(float);   /* write C */
    long long totalBytes = readBytes + writeBytes;
    double time_s = elapsed_ms / 1000.0;
    double measuredBW = totalBytes / time_s / 1e9;

    printf("  Bytes read:      %lld (%.1f MB)\n", readBytes, readBytes / 1e6);
    printf("  Bytes written:   %lld (%.1f MB)\n", writeBytes, writeBytes / 1e6);
    printf("  Total bytes:     %lld (%.1f MB)\n", totalBytes, totalBytes / 1e6);
    printf("  Kernel time:     %.4f ms\n", elapsed_ms);
    printf("  Measured BW:     %.1f GB/s\n", measuredBW);
    printf("  BW Utilization:  %.1f%% of theoretical\n\n", measuredBW / theoreticalBW * 100);

    /* Multi-size bandwidth benchmark */
    printf("--- Bandwidth vs Vector Size ---\n");
    printf("  %12s  %10s  %12s  %12s\n", "N", "Time(ms)", "BW(GB/s)", "Util(%)");

    int sizes[] = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
    for (int s = 0; s < 6; s++) {
        int n = sizes[s];
        size_t b = n * sizeof(float);
        float *da, *db, *dc;
        CUDA_CHECK(cudaMalloc(&da, b));
        CUDA_CHECK(cudaMalloc(&db, b));
        CUDA_CHECK(cudaMalloc(&dc, b));

        /* Init */
        CUDA_CHECK(cudaMemset(da, 0, b));
        CUDA_CHECK(cudaMemset(db, 0, b));

        int blk = (n + THREADS - 1) / THREADS;

        /* Warmup */
        vectorAdd<<<blk, THREADS>>>(da, db, dc, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<blk, THREADS>>>(da, db, dc, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        double bw = 3.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;
        printf("  %12d  %10.4f  %12.1f  %12.1f\n", n, ms, bw, bw / theoreticalBW * 100);

        cudaFree(da); cudaFree(db); cudaFree(dc);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
