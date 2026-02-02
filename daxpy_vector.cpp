#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    int n = 65536;  // 2^16
    vector<double> X(n), Y(n, 2.0);
    double a = 3.5;

    cout << "DAXPY: X[i] = a*X[i] + Y[i]\n";
    cout << "Size: " << n << "\n\n";

    cout << "Threads\tTime(ms)\tSpeedup\n";

    double t1 = 0.0;

    // Test 1, 2, 4, 8, 16 threads
    for (int threads = 1; threads <= omp_get_max_threads(); threads *= 2) {

        // Reset X before each execution
        for (int i = 0; i < n; i++) {
            X[i] = 1.0;
        }

        double start = omp_get_wtime();

        // Run 10000 times for mesureable time
        for (int run = 0; run < 10000; run++) {
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < n; i++) {
                X[i] = a * X[i] + Y[i];
            }
        }

        double time = (omp_get_wtime() - start);  // ms per run

        if (threads == 1) t1 = time;

        cout << threads << "\t" << time << "\t" << t1 / time << "\n";
    }

    return 0;
}
