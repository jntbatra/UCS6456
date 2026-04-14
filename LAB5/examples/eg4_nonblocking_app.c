// Non-blocking: overlap computation with communication
// Process 1 starts receiving, does heavy math concurrently, then waits for data
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARRAY_SIZE 50000000

void do_heavy_computation(int rank) {
    printf("Process %d starting heavy computation...\n", rank);
    double result = 0.0;
    for (long i = 0; i < 500000000; i++) {
        result += sin(i) * cos(i);
    }
    printf("Process %d finished computation. (Result: %f)\n", rank, result);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int* buffer = (int*)malloc(ARRAY_SIZE * sizeof(int));
    MPI_Request request;
    double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) buffer[i] = 1;
        printf("Process 0 initiating background send...\n");
        MPI_Isend(buffer, ARRAY_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        printf("Process 1 initiating background receive...\n");
        MPI_Irecv(buffer, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
        // CPU does heavy math AT THE SAME TIME as network download!
        do_heavy_computation(rank);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        printf("Process 1 confirmed data reception!\n");
    }

    double end_time = MPI_Wtime();
    if (rank == 1) {
        printf("Total Time (Non-Blocking): %f seconds\n", end_time - start_time);
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
