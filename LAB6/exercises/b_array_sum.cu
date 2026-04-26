/*
 * Assignment 6 — Part B: CUDA Array Sum
 * Computes sum of float array elements using parallel reduction.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

#define THREADS 256

/* Reduction kernel: each block reduces its portion to a partial sum */
__global__ void arraySumKernel(const float* input, float* partials, int N)
{
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load element (or zero if out of bounds) */
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    /* Tree reduction in shared memory */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    /* Thread 0 of each block writes the partial sum */
    if (tid == 0)
        partials[blockIdx.x] = sdata[0];
}

/* CPU reference */
float cpu_sum(const float* arr, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += arr[i];
    return (float)sum;
}

static double wall_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

void run_array_sum(int N)
{
    size_t bytes = N * sizeof(float);

    /* 1. Allocate host memory */
    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;  /* each element = 1.0, expected sum = N */

    /* CPU reference */
    double t0 = wall_ms();
    float cpu_result = cpu_sum(h_input, N);
    double cpu_ms = wall_ms() - t0;

    /* 2. Allocate device memory */
    float *d_input, *d_partials, *d_result;
    int threads = THREADS;
    int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaMalloc(&d_input,    bytes));
    CUDA_CHECK(cudaMalloc(&d_partials, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result,   sizeof(float)));

    /* 3. Copy host memory to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    /* CUDA timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    /* 4. First pass: reduce N elements to 'blocks' partial sums */
    arraySumKernel<<<blocks, threads>>>(d_input, d_partials, N);

    /* Second pass: reduce partial sums to single result */
    int remaining = blocks;
    float *d_in = d_partials;
    while (remaining > 1) {
        int next_blocks = (remaining + threads - 1) / threads;
        float *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, next_blocks * sizeof(float)));
        arraySumKernel<<<next_blocks, threads>>>(d_in, d_out, remaining);
        if (d_in != d_partials) cudaFree(d_in);
        d_in = d_out;
        remaining = next_blocks;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    /* 5. Copy results from device to host */
    float gpu_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_result, d_in, sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify */
    float err = fabsf(gpu_result - cpu_result);
    int pass = (err / fabsf(cpu_result + 1e-8f)) < 0.01f;

    printf("  N=%d\n", N);
    printf("  CPU sum = %.1f  (%.3f ms)\n", cpu_result, cpu_ms);
    printf("  GPU sum = %.1f  (%.3f ms)\n", gpu_result, gpu_ms);
    printf("  Speedup = %.1fx\n", cpu_ms / gpu_ms);
    printf("  Relative Error = %.6f  %s\n", err / (fabsf(cpu_result) + 1e-8f),
           pass ? "[PASS]" : "[FAIL]");

    /* 6. Free device memory */
    if (d_in != d_partials) cudaFree(d_in);
    cudaFree(d_input);
    cudaFree(d_partials);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== Assignment 6B: CUDA Array Sum ===\n");
    printf("GPU: %s\n\n", prop.name);

    int sizes[] = {1024, 1 << 16, 1 << 20, 1 << 24};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n_sizes; s++) {
        printf("--- Test %d ---\n", s + 1);
        run_array_sum(sizes[s]);
        printf("\n");
    }

    return 0;
}
