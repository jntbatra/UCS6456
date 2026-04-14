/*
 * Q5: Find all perfect numbers using master-slave MPI pattern
 * A perfect number equals the sum of its proper divisors
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

int is_perfect(int n) {
    if (n < 2) return 0;
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i)
                sum += n / i;
        }
    }
    return sum == n;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    double start_time = MPI_Wtime();

    if (rank == 0) {
        /* MASTER */
        int next_number = 2;
        int perfect_count = 0;
        int perfect_numbers[20];  /* won't find more than 20 below 100k */
        int active_slaves = size - 1;

        /* Wait for initial requests */
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

        while (active_slaves > 0) {
            int result;
            MPI_Status status;
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int slave = status.MPI_SOURCE;

            if (result > 0) {
                perfect_numbers[perfect_count++] = result;
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
        printf("=== Perfect Number Finder (Master-Slave) ===\n");
        printf("Range: 2 to %d | Processes: %d\n", MAX_VALUE, size);
        printf("Perfect numbers found: %d\n", perfect_count);
        printf("Values: ");
        for (int i = 0; i < perfect_count; i++)
            printf("%d ", perfect_numbers[i]);
        printf("\n");
        printf("Time: %.4f s\n", elapsed);

        /* Known perfect numbers below 100000: 6, 28, 496, 8128 */
        printf("Verification: expected {6, 28, 496, 8128}, got %d numbers %s\n",
               perfect_count, perfect_count == 4 ? "[PASS]" : "[FAIL]");

    } else {
        /* SLAVE */
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

        while (1) {
            int number;
            MPI_Status status;
            MPI_Recv(&number, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) break;

            int result = is_perfect(number) ? number : -number;
            MPI_Send(&result, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
