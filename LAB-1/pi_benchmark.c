#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;

double compute_pi_seq(long steps)
{
    double sum = 0.0;
    double step = 1.0 / (double)steps;
    for (long i = 0; i < steps; i++)
    {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

int main(void)
{
    int max_threads = omp_get_max_threads();

    printf("Pi calculation using rectangle rule\n");
    printf("Steps: %ld\n\n", num_steps);

    double start = omp_get_wtime();
    double pi = compute_pi_seq(num_steps);
    double time_seq = omp_get_wtime() - start;
    printf("Sequential: pi = %.12f  Time(s) = %.6f\n\n", pi, time_seq);

    printf("Threads\tTime(s)\tSpeedup\n");
    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
        double sum = 0.0;
        start = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum) num_threads(threads)
        for (long i = 0; i < num_steps; i++)
        {
            double x = (i + 0.5) * (1.0 / (double)num_steps);
            sum += 4.0 / (1.0 + x * x);
        }
        double time = omp_get_wtime() - start;
        printf("%d\t%.6f\t%.2f\n", threads, time, time_seq / time);
    }

    return 0;
}
