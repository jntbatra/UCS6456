// Blocking communication: CPU idle while waiting for data
// Process 0 sends large array to Process 1, which computes after receiving
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
    double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) buffer[i] = 1;
        printf("Process 0 sending massive data...\n");
        MPI_Ssend(buffer, ARRAY_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        printf("Process 1 waiting for data...\n");
        MPI_Recv(buffer, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received data!\n");
        do_heavy_computation(rank);
    }

    double end_time = MPI_Wtime();
    if (rank == 1) {
        printf("Total Time (Blocking): %f seconds\n", end_time - start_time);
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
