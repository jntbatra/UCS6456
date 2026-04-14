/*
 * Q4: Find all primes using master-slave MPI pattern
 * Master distributes numbers, slaves test primality
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 100000
#define TAG_REQUEST 1
#define TAG_RESULT  2
#define TAG_WORK    3
#define TAG_DONE    4

int is_prime(int n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Need at least 2 processes (1 master + 1 slave)\n");
        MPI_Finalize();
        return 1;
    }

    double start_time = MPI_Wtime();

    if (rank == 0) {
        /* MASTER */
        int next_number = 2;
        int primes_found = 0;
        int active_slaves = size - 1;

        /* Wait for initial requests from all slaves */
        for (int i = 0; i < active_slaves; i++) {
            int msg;
            MPI_Status status;
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &status);
            int slave = status.MPI_SOURCE;

            if (next_number <= MAX_VALUE) {
                MPI_Send(&next_number, 1, MPI_INT, slave, TAG_WORK, MPI_COMM_WORLD);
                next_number++;
            }
        }

        /* Process results and send new work */
        while (active_slaves > 0) {
            int result;
            MPI_Status status;
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int slave = status.MPI_SOURCE;

            if (result > 0) {
                primes_found++;
            }

            if (next_number <= MAX_VALUE) {
                MPI_Send(&next_number, 1, MPI_INT, slave, TAG_WORK, MPI_COMM_WORLD);
                next_number++;
            } else {
                int done = -1;
                MPI_Send(&done, 1, MPI_INT, slave, TAG_DONE, MPI_COMM_WORLD);
                active_slaves--;
            }
        }

        double elapsed = MPI_Wtime() - start_time;
        printf("=== Prime Finder (Master-Slave) ===\n");
        printf("Range: 2 to %d | Processes: %d\n", MAX_VALUE, size);
        printf("Primes found: %d\n", primes_found);
        printf("Time: %.4f s\n", elapsed);

        /* Known: there are 9592 primes below 100000 */
        printf("Verification: expected 9592 primes, got %d %s\n",
               primes_found, primes_found == 9592 ? "[PASS]" : "[FAIL]");

    } else {
        /* SLAVE */
        /* Send initial request (zero = just starting) */
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

        while (1) {
            int number;
            MPI_Status status;
            MPI_Recv(&number, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) break;

            /* Test primality and send result */
            int result = is_prime(number) ? number : -number;
            MPI_Send(&result, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
