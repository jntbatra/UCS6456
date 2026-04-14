/*
 * Q3: Distributed Dot Product & Amdahl's Law
 * 500 million elements, MPI_Bcast + MPI_Reduce
 * Test with 1, 2, 4, 8 processes
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOTAL_SIZE 500000000L  /* 500 million */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Broadcast multiplier from rank 0 */
    double multiplier = 2.0;
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Each process computes its chunk locally (no scatter needed) */
    long local_size = TOTAL_SIZE / size;
    long start_idx = rank * local_size;
    /* Last process handles remainder */
    if (rank == size - 1)
        local_size = TOTAL_SIZE - start_idx;

    double *local_A = (double*)malloc(local_size * sizeof(double));
    double *local_B = (double*)malloc(local_size * sizeof(double));
    if (!local_A || !local_B) {
        fprintf(stderr, "Rank %d: malloc failed for %ld elements\n", rank, local_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Generate local chunks: A[i] = 1.0, B[i] = 2.0 * multiplier */
    for (long i = 0; i < local_size; i++) {
        local_A[i] = 1.0;
        local_B[i] = multiplier;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* Local dot product */
    double local_dot = 0.0;
    for (long i = 0; i < local_size; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    /* Global reduction */
    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double elapsed = end_time - start_time;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double expected = 1.0 * multiplier * TOTAL_SIZE;
        printf("=== Distributed Dot Product ===\n");
        printf("Vector size: %ld | Processes: %d\n", TOTAL_SIZE, size);
        printf("Multiplier: %.1f\n", multiplier);
        printf("Result:   %.0f\n", global_dot);
        printf("Expected: %.0f\n", expected);
        printf("Time:     %.6f s\n", max_elapsed);
        printf("Verification: %s\n",
               fabs(global_dot - expected) < 1e-6 ? "[PASS]" : "[FAIL]");
    }

    free(local_A);
    free(local_B);
    MPI_Finalize();
    return 0;
}
