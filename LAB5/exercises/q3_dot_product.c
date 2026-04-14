/*
 * Q3: Distributed Dot Product with Amdahl's Law
 * 500 million elements, local generation, MPI_Bcast + MPI_Reduce
 * Test with 1, 2, 4, 8 processes — calculate speedup and efficiency
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOTAL_SIZE 500000000L

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Step 1: Broadcast multiplier */
    double multiplier = 2.0;
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Step 2: Local generation — each process makes its own chunk */
    long local_size = TOTAL_SIZE / size;
    long start_idx = rank * local_size;
    if (rank == size - 1)
        local_size = TOTAL_SIZE - start_idx;

    double *A = (double*)malloc(local_size * sizeof(double));
    double *B = (double*)malloc(local_size * sizeof(double));
    if (!A || !B) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < local_size; i++) {
        A[i] = 1.0;
        B[i] = multiplier;
    }

    /* Step 3: Timed computation */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double local_dot = 0.0;
    for (long i = 0; i < local_size; i++)
        local_dot += A[i] * B[i];

    double t_compute = MPI_Wtime() - t_start;

    /* Step 4: Global reduction */
    double t_comm_start = MPI_Wtime();
    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_comm = MPI_Wtime() - t_comm_start;
    double t_total = t_compute + t_comm;

    /* Gather max times */
    double max_compute, max_comm, max_total;
    MPI_Reduce(&t_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double expected = 1.0 * multiplier * TOTAL_SIZE;
        printf("=== Distributed Dot Product (Amdahl's Law) ===\n");
        printf("Vector size: %ld | Processes: %d\n", TOTAL_SIZE, size);
        printf("Result:   %.0f (expected %.0f) %s\n", global_dot, expected,
               fabs(global_dot - expected) < 1.0 ? "[PASS]" : "[FAIL]");
        printf("Compute time: %.6f s\n", max_compute);
        printf("Comm time:    %.6f s\n", max_comm);
        printf("Total time:   %.6f s\n", max_total);
        printf("Comm overhead: %.1f%%\n", max_comm / max_total * 100.0);
    }

    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}
