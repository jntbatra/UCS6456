/*
 * Q2: Blocking vs Non-Blocking Communication Benchmark
 * Compares total time when Process 1 must receive then compute (blocking)
 * vs receive + compute concurrently (non-blocking)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARRAY_SIZE 50000000  /* 50M ints ~ 200 MB */
#define COMPUTE_ITERS 500000000L

double do_computation(int rank) {
    double result = 0.0;
    for (long i = 0; i < COMPUTE_ITERS; i++) {
        result += sin((double)i) * cos((double)i);
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) printf("This benchmark requires exactly 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    int *buffer = (int*)malloc(ARRAY_SIZE * sizeof(int));

    /* ===== BLOCKING version ===== */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_block_start = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) buffer[i] = i;
        MPI_Ssend(buffer, ARRAY_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        /* Must wait for data, THEN compute — sequential */
        MPI_Recv(buffer, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        do_computation(rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_block = MPI_Wtime() - t_block_start;

    /* ===== NON-BLOCKING version ===== */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_nonblock_start = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) buffer[i] = i;
        MPI_Request req;
        MPI_Isend(buffer, ARRAY_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {
        MPI_Request req;
        /* Start receiving in background */
        MPI_Irecv(buffer, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
        /* Compute while data transfers */
        do_computation(rank);
        /* Ensure data arrived before using it */
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_nonblock = MPI_Wtime() - t_nonblock_start;

    if (rank == 0) {
        printf("=== Blocking vs Non-Blocking Benchmark ===\n");
        printf("Array: %d ints (~%d MB) | Compute: %ldM iterations\n",
               ARRAY_SIZE, (int)(ARRAY_SIZE * sizeof(int) / 1000000), COMPUTE_ITERS / 1000000);
        printf("Blocking time:     %.4f s\n", t_block);
        printf("Non-blocking time: %.4f s\n", t_nonblock);
        printf("Speedup:           %.2fx\n", t_block / t_nonblock);
        printf("Time saved:        %.4f s (%.1f%%)\n",
               t_block - t_nonblock,
               (t_block - t_nonblock) / t_block * 100.0);
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
