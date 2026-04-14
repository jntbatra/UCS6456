// Non-blocking ring communication — no deadlock even with massive arrays
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000  // 10 Million — no deadlock!

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

    MPI_Request requests[2];

    printf("Process %d initiating background transfers...\n", rank);

    MPI_Irecv(recv_buf, ARRAY_SIZE, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(send_buf, ARRAY_SIZE, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, &requests[1]);

    printf("Process %d is doing math while data transfers...\n", rank);

    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    printf("Process %d successfully completed all transfers!\n", rank);

    free(send_buf);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}
