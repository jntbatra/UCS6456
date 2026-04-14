/*
 * Q1: Broadcast Race — MyBcast (linear MPI_Send) vs MPI_Bcast (tree)
 * 10 million doubles (~80 MB), test with 2, 4, 8, 16 processes
 * Demonstrates why collective ops beat manual point-to-point
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE 10000000  /* 10 million doubles ~ 80 MB */
#define REPEATS 5

/* Part A: MyBcast — linear for-loop of MPI_Send from rank 0 */
void my_bcast(double *data, int count, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        for (int i = 0; i < size; i++) {
            if (i != root)
                MPI_Send(data, count, MPI_DOUBLE, i, 0, comm);
        }
    } else {
        MPI_Recv(data, count, MPI_DOUBLE, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *data = (double*)malloc(ARRAY_SIZE * sizeof(double));
    if (!data) { MPI_Abort(MPI_COMM_WORLD, 1); }

    /* === Part A: MyBcast timing === */
    double my_best = 1e30;
    for (int r = 0; r < REPEATS; r++) {
        if (rank == 0)
            for (int i = 0; i < ARRAY_SIZE; i++) data[i] = (double)i;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        my_bcast(data, ARRAY_SIZE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - t0;
        double max_t;
        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && max_t < my_best) my_best = max_t;
    }

    /* === Part B: MPI_Bcast timing === */
    double mpi_best = 1e30;
    for (int r = 0; r < REPEATS; r++) {
        if (rank == 0)
            for (int i = 0; i < ARRAY_SIZE; i++) data[i] = (double)i;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Bcast(data, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - t0;
        double max_t;
        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && max_t < mpi_best) mpi_best = max_t;
    }

    if (rank == 0) {
        printf("=== Broadcast Race (%.0f MB) ===\n", ARRAY_SIZE * sizeof(double) / 1e6);
        printf("Processes: %d | Repeats: %d (best of)\n", size, REPEATS);
        printf("MyBcast (linear):  %.6f s\n", my_best);
        printf("MPI_Bcast (tree):  %.6f s\n", mpi_best);
        printf("MPI_Bcast advantage: %.2fx faster\n", my_best / mpi_best);

        /* Verify */
        int pass = 1;
        for (int i = 0; i < 10; i++)
            if (data[i] != (double)i) { pass = 0; break; }
        printf("Verification: %s\n", pass ? "[PASS]" : "[FAIL]");
    }

    /* Verify on non-root too */
    int local_pass = 1;
    for (int i = 0; i < 10; i++)
        if (data[i] != (double)i) { local_pass = 0; break; }
    int all_pass;
    MPI_Reduce(&local_pass, &all_pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("All ranks verified: %s\n", all_pass ? "[PASS]" : "[FAIL]");

    free(data);
    MPI_Finalize();
    return 0;
}
