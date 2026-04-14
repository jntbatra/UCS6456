#include <cmath>
#include <vector>
#include <omp.h>
#include <immintrin.h>
#include <cstring>

// Version 1: Sequential baseline (double precision)
void correlate_sequential(int ny, int nx, const float* data, float* result) {
    // Normalize each row: mean=0, std=1
    std::vector<double> norm(ny * nx);

    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++) {
            sum += data[x + y * nx];
        }
        double mean = sum / nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double v = data[x + y * nx] - mean;
            norm[x + y * nx] = v;
            sq_sum += v * v;
        }
        double inv_std = 1.0 / std::sqrt(sq_sum);
        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] *= inv_std;
        }
    }

    // Compute correlation: dot product of normalized rows
    for (int i = 0; i < ny; i++) {
        for (int j = i; j < ny; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++) {
                dot += norm[x + i * nx] * norm[x + j * nx];
            }
            result[i + j * ny] = (float)dot;
        }
    }
}

// Version 2: OpenMP parallelized
void correlate_parallel(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm(ny * nx);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++) {
            sum += data[x + y * nx];
        }
        double mean = sum / nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double v = data[x + y * nx] - mean;
            norm[x + y * nx] = v;
            sq_sum += v * v;
        }
        double inv_std = 1.0 / std::sqrt(sq_sum);
        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] *= inv_std;
        }
    }

    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < ny; j++) {
            if (j >= i) {
                double dot = 0.0;
                for (int x = 0; x < nx; x++) {
                    dot += norm[x + i * nx] * norm[x + j * nx];
                }
                result[i + j * ny] = (float)dot;
            }
        }
    }
}

// Version 3: Fully optimized — OpenMP + vectorization + cache blocking + padded layout
void correlate_optimized(int ny, int nx, const float* data, float* result) {
    // Pad columns to multiple of 8 for AVX alignment
    int pad_nx = (nx + 3) & ~3; // align to 4 doubles

    // Allocate aligned, padded, normalized data
    std::vector<double> norm(ny * pad_nx, 0.0);

    // Normalize rows in parallel
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        for (int x = 0; x < nx; x++) {
            sum += data[x + y * nx];
        }
        double mean = sum / nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double v = data[x + y * nx] - mean;
            norm[x + y * pad_nx] = v;
            sq_sum += v * v;
        }

        double inv_std = (sq_sum > 0.0) ? 1.0 / std::sqrt(sq_sum) : 0.0;
        for (int x = 0; x < nx; x++) {
            norm[x + y * pad_nx] *= inv_std;
        }
        // Padding already zeroed
    }

    // Compute correlation with blocking for cache locality
    constexpr int BLOCK = 8;
    int nblocks = (ny + BLOCK - 1) / BLOCK;

    #pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (int bi = 0; bi < nblocks; bi++) {
        for (int bj = 0; bj < nblocks; bj++) {
            if (bj >= bi) {
                int i_start = bi * BLOCK;
                int j_start = bj * BLOCK;
                int i_end = std::min(i_start + BLOCK, ny);
                int j_end = std::min(j_start + BLOCK, ny);

                for (int i = i_start; i < i_end; i++) {
                    int j_lo = (bi == bj) ? i : j_start;
                    for (int j = j_lo; j < j_end; j++) {
                        const double* row_i = &norm[i * pad_nx];
                        const double* row_j = &norm[j * pad_nx];

                        // Vectorized dot product using AVX2
                        __m256d sum0 = _mm256_setzero_pd();
                        __m256d sum1 = _mm256_setzero_pd();

                        int x = 0;
                        for (; x + 7 < nx; x += 8) {
                            __m256d a0 = _mm256_loadu_pd(row_i + x);
                            __m256d b0 = _mm256_loadu_pd(row_j + x);
                            sum0 = _mm256_fmadd_pd(a0, b0, sum0);

                            __m256d a1 = _mm256_loadu_pd(row_i + x + 4);
                            __m256d b1 = _mm256_loadu_pd(row_j + x + 4);
                            sum1 = _mm256_fmadd_pd(a1, b1, sum1);
                        }

                        sum0 = _mm256_add_pd(sum0, sum1);

                        // Horizontal sum
                        __m128d lo = _mm256_castpd256_pd128(sum0);
                        __m128d hi = _mm256_extractf128_pd(sum0, 1);
                        lo = _mm_add_pd(lo, hi);
                        double tmp[2];
                        _mm_storeu_pd(tmp, lo);
                        double dot = tmp[0] + tmp[1];

                        // Handle remaining elements
                        for (; x < nx; x++) {
                            dot += row_i[x] * row_j[x];
                        }

                        result[i + j * ny] = (float)dot;
                    }
                }
            }
        }
    }
}

// Dispatcher
void correlate(int ny, int nx, const float* data, float* result, int version) {
    switch (version) {
        case 1: correlate_sequential(ny, nx, data, result); break;
        case 2: correlate_parallel(ny, nx, data, result); break;
        case 3: correlate_optimized(ny, nx, data, result); break;
        default: correlate_optimized(ny, nx, data, result); break;
    }
}
