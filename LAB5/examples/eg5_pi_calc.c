// Calculate Pi using MPI_Bcast + MPI_Reduce
// Each process computes partial area under curve, reduce sums them
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    long num_steps = 1000000;
    double step, x, sum = 0.0, pi, total_pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Process 0 broadcasting total steps: %ld\n", num_steps);
    }
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;
    for (int i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;

    printf("Process %d calculated a partial Pi value of: %f\n", rank, pi);

    MPI_Reduce(&pi, &total_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("========================================\n");
        printf("Final Pi calculated by Rank 0: %.10f\n", total_pi);
    }

    MPI_Finalize();
    return 0;
}
