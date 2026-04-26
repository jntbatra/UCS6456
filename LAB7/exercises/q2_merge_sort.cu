/*
 * Assignment 7 — Problem 2: Merge Sort
 * (a) Pipelined parallelization approach
 * (b) Parallel merge sort using CUDA
 * (c) Performance comparison
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define ARRAY_SIZE 1000

static double wall_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* CPU merge for two sorted sub-arrays */
void cpu_merge(int* arr, int* temp, int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= mid)  temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    memcpy(&arr[left], &temp[left], (right - left + 1) * sizeof(int));
}

/* (a) CPU pipelined merge sort — bottom-up iterative */
void cpu_merge_sort(int* arr, int n)
{
    int* temp = (int*)malloc(n * sizeof(int));
    for (int width = 1; width < n; width *= 2) {
        for (int left = 0; left < n; left += 2 * width) {
            int mid = left + width - 1;
            int right = left + 2 * width - 1;
            if (mid >= n) mid = n - 1;
            if (right >= n) right = n - 1;
            if (mid < right)
                cpu_merge(arr, temp, left, mid, right);
        }
    }
    free(temp);
}

/* GPU merge kernel: each thread merges one pair of sub-arrays */
__device__ void device_merge(int* arr, int* temp, int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= mid)  temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int x = left; x <= right; x++)
        arr[x] = temp[x];
}

__global__ void mergeSortKernel(int* arr, int* temp, int n, int width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * 2 * width;
    if (left >= n) return;

    int mid = left + width - 1;
    int right = left + 2 * width - 1;
    if (mid >= n) mid = n - 1;
    if (right >= n) right = n - 1;

    if (mid < right)
        device_merge(arr, temp, left, mid, right);
}

int is_sorted(const int* arr, int n)
{
    for (int i = 1; i < n; i++)
        if (arr[i] < arr[i-1]) return 0;
    return 1;
}

int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== Assignment 7 — Problem 2: Merge Sort (n=%d) ===\n", ARRAY_SIZE);
    printf("GPU: %s\n\n", prop.name);

    /* Generate random array */
    srand(42);
    int *original = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++)
        original[i] = rand() % 10000;

    /* (a) CPU pipelined merge sort */
    int *cpu_arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    memcpy(cpu_arr, original, ARRAY_SIZE * sizeof(int));

    double t0 = wall_ms();
    cpu_merge_sort(cpu_arr, ARRAY_SIZE);
    double cpu_ms = wall_ms() - t0;

    printf("(a) CPU Pipelined Merge Sort:\n");
    printf("    Time: %.4f ms   Sorted: %s\n\n",
           cpu_ms, is_sorted(cpu_arr, ARRAY_SIZE) ? "[PASS]" : "[FAIL]");

    /* (b) GPU parallel merge sort */
    int *d_arr, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_arr,  ARRAY_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp, ARRAY_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, original, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int width = 1; width < ARRAY_SIZE; width *= 2) {
        int pairs = (ARRAY_SIZE + 2 * width - 1) / (2 * width);
        int threads = 256;
        int blocks = (pairs + threads - 1) / threads;
        mergeSortKernel<<<blocks, threads>>>(d_arr, d_temp, ARRAY_SIZE, width);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    int *gpu_arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    CUDA_CHECK(cudaMemcpy(gpu_arr, d_arr, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    printf("(b) GPU Parallel Merge Sort:\n");
    printf("    Time: %.4f ms   Sorted: %s\n\n",
           gpu_ms, is_sorted(gpu_arr, ARRAY_SIZE) ? "[PASS]" : "[FAIL]");

    /* Verify both produce same result */
    int match = 1;
    for (int i = 0; i < ARRAY_SIZE; i++)
        if (cpu_arr[i] != gpu_arr[i]) { match = 0; break; }

    /* (c) Performance comparison */
    printf("(c) Performance Comparison:\n");
    printf("    CPU: %.4f ms\n", cpu_ms);
    printf("    GPU: %.4f ms\n", gpu_ms);
    printf("    Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("    Results match: %s\n", match ? "[PASS]" : "[FAIL]");

    /* Benchmark larger sizes */
    printf("\n--- Scaling Benchmark ---\n");
    printf("  %10s  %12s  %12s  %10s\n", "N", "CPU(ms)", "GPU(ms)", "Speedup");

    int test_sizes[] = {100, 500, 1000, 5000, 10000, 50000};
    for (int t = 0; t < 6; t++) {
        int n = test_sizes[t];
        int *h_arr = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) h_arr[i] = rand() % 100000;

        /* CPU */
        int *h_cpu = (int*)malloc(n * sizeof(int));
        memcpy(h_cpu, h_arr, n * sizeof(int));
        t0 = wall_ms();
        cpu_merge_sort(h_cpu, n);
        double c_ms = wall_ms() - t0;

        /* GPU */
        int *da, *dt;
        CUDA_CHECK(cudaMalloc(&da, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dt, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(da, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        for (int w = 1; w < n; w *= 2) {
            int pairs = (n + 2 * w - 1) / (2 * w);
            int thr = 256, blk = (pairs + thr - 1) / thr;
            mergeSortKernel<<<blk, thr>>>(da, dt, n, w);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float g_ms;
        CUDA_CHECK(cudaEventElapsedTime(&g_ms, start, stop));

        printf("  %10d  %12.4f  %12.4f  %10.2f\n", n, c_ms, g_ms, c_ms / g_ms);

        cudaFree(da); cudaFree(dt);
        free(h_arr); free(h_cpu);
    }

    cudaFree(d_arr); cudaFree(d_temp);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(original); free(cpu_arr); free(gpu_arr);
    return 0;
}
