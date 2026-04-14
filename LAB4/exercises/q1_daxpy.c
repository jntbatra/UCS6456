/*
 * Q1: DAXPY with MPI
 * X[i] = a*X[i] + Y[i], vectors of size 2^16
 * Measure speedup of MPI vs sequential
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE (1 << 16)  /* 65536 */
#define REPEATS 50

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = VECTOR_SIZE / size;
    double a = 2.5;

    double *X = (double*)malloc(VECTOR_SIZE * sizeof(double));
    double *Y = (double*)malloc(VECTOR_SIZE * sizeof(double));
    double *local_X = (double*)malloc(local_n * sizeof(double));
    double *local_Y = (double*)malloc(local_n * sizeof(double));

    /* Initialize on rank 0 */
    if (rank == 0) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
            X[i] = (double)(i % 1000) / 1000.0;
            Y[i] = (double)((i * 7) % 1000) / 1000.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Sequential timing on rank 0 --- */
    double t_seq = 0.0;
    if (rank == 0) {
        double *X_copy = (double*)malloc(VECTOR_SIZE * sizeof(double));
        double start = MPI_Wtime();
        for (int r = 0; r < REPEATS; r++) {
            for (int i = 0; i < VECTOR_SIZE; i++)
                X_copy[i] = X[i];
            for (int i = 0; i < VECTOR_SIZE; i++)
                X_copy[i] = a * X_copy[i] + Y[i];
        }
        t_seq = (MPI_Wtime() - start) / REPEATS;
        free(X_copy);
    }

    /* --- Parallel DAXPY --- */
    double t_par_total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        /* Re-init X each repeat */
        if (rank == 0) {
            for (int i = 0; i < VECTOR_SIZE; i++)
                X[i] = (double)(i % 1000) / 1000.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        /* Distribute */
        MPI_Scatter(X, local_n, MPI_DOUBLE, local_X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(Y, local_n, MPI_DOUBLE, local_Y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Compute DAXPY locally */
        for (int i = 0; i < local_n; i++) {
            local_X[i] = a * local_X[i] + local_Y[i];
        }

        /* Gather results */
        MPI_Gather(local_X, local_n, MPI_DOUBLE, X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - start;

        double max_elapsed;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) t_par_total += max_elapsed;
    }

    if (rank == 0) {
        double t_par = t_par_total / REPEATS;
        double speedup = t_seq / t_par;
        double efficiency = speedup / size * 100.0;

        printf("=== DAXPY MPI Benchmark ===\n");
        printf("Vector size: %d | Processes: %d | Repeats: %d\n", VECTOR_SIZE, size, REPEATS);
        printf("Sequential time:  %.6f s\n", t_seq);
        printf("Parallel time:    %.6f s\n", t_par);
        printf("Speedup:          %.2fx\n", speedup);
        printf("Efficiency:       %.1f%%\n", efficiency);

        /* Verify correctness */
        double max_err = 0.0;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            double x_orig = (double)(i % 1000) / 1000.0;
            double y_orig = (double)((i * 7) % 1000) / 1000.0;
            double expected = a * x_orig + y_orig;
            double err = fabs(X[i] - expected);
            if (err > max_err) max_err = err;
        }
        printf("Verification: max error = %.2e %s\n", max_err, max_err < 1e-10 ? "[PASS]" : "[FAIL]");
    }

    free(X); free(Y); free(local_X); free(local_Y);
    MPI_Finalize();
    return 0;
}
