#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
int main()
{
    char *env = getenv("OMP_NUM_THREADS");
    int num_threads = env ? atoi(env) : 4;
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
        printf("Thread ID: %d\n", omp_get_thread_num());
    }
    return 0;
} 