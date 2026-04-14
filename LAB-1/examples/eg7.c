#include <stdio.h>
#include <omp.h>

int main() {
    int counter = 0;

    #pragma omp parallel
    {
        #pragma omp critical
        {
            counter++;
        }
    }

    printf("Counter = %d\n", counter);
    return 0;
}
