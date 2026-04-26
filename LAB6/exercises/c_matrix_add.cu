/*
 * Assignment 6 — Part C: CUDA Matrix Addition
 * Adds two large integer matrices with performance analysis.
 */
#include <stdio.h>
#include <stdlib.h>
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

#define BLOCK_SIZE 16

/* Matrix addition kernel: C = A + B */
__global__ void matAddKernel(const int* A, const int* B, int* C, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

/* CPU reference */
void cpu_matAdd(const int* A, const int* B, int* C, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        C[i] = A[i] + B[i];
}

static double wall_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

void run_matrix_add(int rows, int cols)
{
    int N = rows * cols;
    size_t bytes = N * sizeof(int);

    /* Allocate host */
    int *h_A   = (int*)malloc(bytes);
    int *h_B   = (int*)malloc(bytes);
    int *h_C   = (int*)malloc(bytes);
    int *h_ref = (int*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    /* CPU reference */
    double t0 = wall_ms();
    cpu_matAdd(h_A, h_B, h_ref, rows, cols);
    double cpu_ms = wall_ms() - t0;

    /* Allocate device */
    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch 2D grid */
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_ref[i]) { pass = 0; break; }
    }

    long long flops = (long long)N;  /* one addition per element */
    long long reads = 2LL * N;       /* read A[i] and B[i] */
    long long writes = (long long)N; /* write C[i] */

    printf("  Matrix: %d x %d (%d elements)\n", rows, cols, N);
    printf("  CPU: %.3f ms   GPU: %.3f ms   Speedup: %.1fx\n",
           cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    printf("  Floating-point ops:    %lld (one add per element)\n", flops);
    printf("  Global memory reads:   %lld (%lld bytes)\n", reads, reads * (long long)sizeof(int));
    printf("  Global memory writes:  %lld (%lld bytes)\n", writes, writes * (long long)sizeof(int));
    printf("  Effective BW:          %.1f GB/s\n",
           (reads + writes) * sizeof(int) / (gpu_ms / 1000.0) / 1e9);
    printf("  Verification: %s\n", pass ? "[PASS]" : "[FAIL]");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C); free(h_ref);
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== Assignment 6C: CUDA Matrix Addition ===\n");
    printf("GPU: %s\n\n", prop.name);

    int sizes[][2] = {{512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n_sizes; s++) {
        printf("--- Test %d ---\n", s + 1);
        run_matrix_add(sizes[s][0], sizes[s][1]);
        printf("\n");
    }

    return 0;
}
