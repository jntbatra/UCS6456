// Blocking Send and Receive: ring communication
// Each process sends to right neighbor, receives from left neighbor
// Small ARRAY_SIZE works fine; large (10M) can deadlock
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    int* send_buf = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int* recv_buf = (int*)malloc(ARRAY_SIZE * sizeof(int));

    for (int i = 0; i < ARRAY_SIZE; i++)
        send_buf[i] = rank * 1000 + i;

    printf("Process %d sending data to %d...\n", rank, right_neighbor);

    MPI_Send(send_buf, ARRAY_SIZE, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD);
    MPI_Recv(recv_buf, ARRAY_SIZE, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d successfully received data from %d!\n", rank, left_neighbor);

    free(send_buf);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}
