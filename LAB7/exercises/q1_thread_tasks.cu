/*
 * Assignment 7 — Problem 1: Different Thread Tasks
 * Task A: Sum first N integers iteratively (no formula)
 * Task B: Sum first N integers using direct formula n*(n+1)/2
 */
#include <stdio.h>
#include <stdlib.h>
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

#define N 1024

/* Task A: Iterative sum — each thread computes sum(1..N) iteratively */
__global__ void sumIterative(const int* input, long long* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        long long sum = 0;
        for (int i = 0; i < n; i++)
            sum += input[i];
        output[0] = sum;
    }
}

/* Task B: Formula sum — each thread uses n*(n+1)/2 */
__global__ void sumFormula(long long* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        output[0] = (long long)n * (n + 1) / 2;
    }
}

/* Multi-thread iterative: parallel reduction-based sum */
__global__ void sumIterativeParallel(const int* input, long long* output, int n)
{
    __shared__ long long sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? (long long)input[i] : 0LL;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd((unsigned long long*)output, (unsigned long long)sdata[0]);
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== Assignment 7 — Problem 1: Different Thread Tasks ===\n");
    printf("GPU: %s  N=%d\n\n", prop.name, N);

    /* Step 1-2: Create input and output arrays */
    int *h_input = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        h_input[i] = i + 1;  /* Step 4: Fill with first N integers */

    long long expected = (long long)N * (N + 1) / 2;

    /* Step 3: Allocate device memory */
    int *d_input;
    long long *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(long long)));

    /* Step 5: Copy host to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    /* CUDA timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    /* --- Task A: Iterative sum (single thread) --- */
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(long long)));
    CUDA_CHECK(cudaEventRecord(start));
    sumIterative<<<1, 1>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    long long result_a;
    CUDA_CHECK(cudaMemcpy(&result_a, d_output, sizeof(long long), cudaMemcpyDeviceToHost));
    printf("Task A (Iterative, 1 thread): sum=%lld  expected=%lld  %s  time=%.3f ms\n",
           result_a, expected, result_a == expected ? "[PASS]" : "[FAIL]", ms);

    /* --- Task A: Iterative sum (parallel reduction) --- */
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(long long)));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    CUDA_CHECK(cudaEventRecord(start));
    sumIterativeParallel<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    long long result_ap;
    CUDA_CHECK(cudaMemcpy(&result_ap, d_output, sizeof(long long), cudaMemcpyDeviceToHost));
    printf("Task A (Iterative, parallel): sum=%lld  expected=%lld  %s  time=%.3f ms\n",
           result_ap, expected, result_ap == expected ? "[PASS]" : "[FAIL]", ms);

    /* --- Task B: Formula sum --- */
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(long long)));
    CUDA_CHECK(cudaEventRecord(start));
    sumFormula<<<1, 1>>>(d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    long long result_b;
    CUDA_CHECK(cudaMemcpy(&result_b, d_output, sizeof(long long), cudaMemcpyDeviceToHost));
    printf("Task B (Formula):             sum=%lld  expected=%lld  %s  time=%.3f ms\n",
           result_b, expected, result_b == expected ? "[PASS]" : "[FAIL]", ms);

    /* Benchmark: compare approaches for various N */
    printf("\n--- Benchmark: Iterative vs Formula vs Parallel ---\n");
    printf("  %10s  %12s  %12s  %12s\n", "N", "Iter(ms)", "Formula(ms)", "Parallel(ms)");

    int test_sizes[] = {256, 1024, 4096, 16384, 65536};
    for (int t = 0; t < 5; t++) {
        int n = test_sizes[t];
        int *h_in = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) h_in[i] = i + 1;

        int *d_in;
        long long *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(long long)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice));

        float iter_ms, form_ms, par_ms;

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(long long)));
        CUDA_CHECK(cudaEventRecord(start));
        sumIterative<<<1, 1>>>(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start, stop));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(long long)));
        CUDA_CHECK(cudaEventRecord(start));
        sumFormula<<<1, 1>>>(d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&form_ms, start, stop));

        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(long long)));
        int blks = (n + 255) / 256;
        CUDA_CHECK(cudaEventRecord(start));
        sumIterativeParallel<<<blks, 256>>>(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&par_ms, start, stop));

        printf("  %10d  %12.4f  %12.4f  %12.4f\n", n, iter_ms, form_ms, par_ms);

        cudaFree(d_in); cudaFree(d_out); free(h_in);
    }

    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_input);
    return 0;
}
