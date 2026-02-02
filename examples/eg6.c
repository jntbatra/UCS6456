#include <stdio.h>
#include <omp.h>
#include <unistd.h>   // for usleep()

#define N 40

void heavy_work(int i) {
    // Make workload unequal
    if (i % 5 == 0)
        usleep(200000);   // 200 ms (heavy)
    else
        usleep(50000);    // 50 ms (light)
}

int main() {
    int i;
    double start, end;

    omp_set_num_threads(16);

    /* ================= STATIC ================= */
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static,4)
    for (i = 0; i < N; i++) {
        heavy_work(i);
    }
    end = omp_get_wtime();
    printf("Static scheduling time : %.3f seconds\n", end - start);

    /* ================= DYNAMIC ================= */
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,2)
    for (i = 0; i < N; i++) {
        heavy_work(i);
    }
    end = omp_get_wtime();
    printf("Dynamic scheduling time: %.3f seconds\n", end - start);

    /* ================= GUIDED ================= */
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (i = 0; i < N; i++) {
        heavy_work(i);
    }
    end = omp_get_wtime();
    printf("Guided scheduling time : %.3f seconds\n", end - start);

    return 0;
}
