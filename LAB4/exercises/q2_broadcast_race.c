/*
 * Q2: Broadcast Race — Linear (MyBcast) vs Tree (MPI_Bcast)
 * 10 million doubles (~80MB), test with 2,4,8,16 processes
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000  /* 10 million doubles */
#define REPEATS 5

/* MyBcast: Rank 0 sends to each rank one-by-one (linear) */
void my_bcast(double *data, int count, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        for (int i = 0; i < size; i++) {
            if (i != root) {
                MPI_Send(data, count, MPI_DOUBLE, i, 0, comm);
            }
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
    if (!data) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialize on rank 0 */
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++)
            data[i] = (double)i;
    }

    /* --- MyBcast (linear) timing --- */
    double my_total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        if (rank == 0) {
            for (int i = 0; i < ARRAY_SIZE; i++)
                data[i] = (double)i;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        my_bcast(data, ARRAY_SIZE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - start;
        double max_elapsed;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) my_total += max_elapsed;
    }

    /* --- MPI_Bcast (tree) timing --- */
    double mpi_total = 0.0;
    for (int r = 0; r < REPEATS; r++) {
        if (rank == 0) {
            for (int i = 0; i < ARRAY_SIZE; i++)
                data[i] = (double)i;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        MPI_Bcast(data, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - start;
        double max_elapsed;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) mpi_total += max_elapsed;
    }

    if (rank == 0) {
        double t_my = my_total / REPEATS;
        double t_mpi = mpi_total / REPEATS;
        printf("=== Broadcast Race (%.0f MB) ===\n", ARRAY_SIZE * sizeof(double) / 1e6);
        printf("Processes: %d | Repeats: %d\n", size, REPEATS);
        printf("MyBcast (linear):  %.6f s\n", t_my);
        printf("MPI_Bcast (tree):  %.6f s\n", t_mpi);
        printf("Speedup (MPI/My):  %.2fx\n", t_my / t_mpi);

        /* Verify last element */
        double expected = (double)(ARRAY_SIZE - 1);
        printf("Verification: data[last] = %.0f (expected %.0f) %s\n",
               data[ARRAY_SIZE-1], expected,
               data[ARRAY_SIZE-1] == expected ? "[PASS]" : "[FAIL]");
    }

    free(data);
    MPI_Finalize();
    return 0;
}
