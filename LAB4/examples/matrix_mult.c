#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rows_per_proc = N / size;

    int matrix[N][N];
    int vector[N];
    int result[N];
    int local_matrix[rows_per_proc][N];
    int local_result[rows_per_proc];

    if (rank == 0) {
        printf("Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = i + j;
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("\nVector:\n");
        for (int i = 0; i < N; i++) {
            vector[i] = i;
            printf("%d ", vector[i]);
        }
        printf("\n\n");
    }

    MPI_Scatter(matrix, rows_per_proc * N, MPI_INT,
                local_matrix, rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; i++) {
        local_result[i] = 0;
        for (int j = 0; j < N; j++) {
            local_result[i] += local_matrix[i][j] * vector[j];
        }
    }

    MPI_Gather(local_result, rows_per_proc, MPI_INT,
               result, rows_per_proc, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", result[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
