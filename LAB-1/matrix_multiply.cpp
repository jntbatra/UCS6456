#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm> // for std::fill

using namespace std;

vector<double> transpose(const vector<double> &mat, int n)
{
    vector<double> t(n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            t[j * n + i] = mat[i * n + j];
        }
    }
    return t;
}

int main()
{
    const int N = 1000;
    vector<double> A(N * N), B(N * N), C(N * N);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    cout << "Matrix Multiply: C = A * B\n";
    cout << "Size: " << N << "x" << N << "\n\n";

    int max_threads = omp_get_max_threads();

    // Simple 1D Version
    cout << "Simple 1D Version:\n";
    cout << "Threads\tTime(s)\tSpeedup\n";
    double t1_simple_1d = 0.0;
    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
        fill(C.begin(), C.end(), 0.0);
        double start = omp_get_wtime();
        for (int run = 0; run < 1; run++)
        {
#pragma omp parallel for num_threads(threads)
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++)
                    {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
        double time = omp_get_wtime() - start;
        if (threads == 1)
            t1_simple_1d = time;
        cout << threads << "\t" << time << "\t" << t1_simple_1d / time << "\n";
    }

    // Transpose 1D Version
    vector<double> B_trans = transpose(B, N);
    cout << "\nTranspose 1D Version:\n";
    cout << "Threads\tTime(s)\tSpeedup\n";
    double t1_trans_1d = 0.0;
    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
        fill(C.begin(), C.end(), 0.0);
        double start = omp_get_wtime();
        for (int run = 0; run < 1; run++)
        {
#pragma omp parallel for num_threads(threads)
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++)
                    {
                        sum += A[i * N + k] * B_trans[j * N + k];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
        double time = omp_get_wtime() - start;
        if (threads == 1)
            t1_trans_1d = time;
        cout << threads << "\t" << time << "\t" << t1_trans_1d / time << "\n";
    }

    // 2D Collapse Version
    cout << "\n2D Collapse Version:\n";
    cout << "Threads\tTime(s)\tSpeedup\n";
    double t1_2d_collapse = 0.0;
    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
        fill(C.begin(), C.end(), 0.0);
        double start = omp_get_wtime();
        for (int run = 0; run < 1; run++)
        {
#pragma omp parallel for num_threads(threads) collapse(2)
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                { 
                    double sum = 0.0;
                    for (int k = 0; k < N; k++)
                    {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
        double time = omp_get_wtime() - start;
        if (threads == 1)
            t1_2d_collapse = time;
        cout << threads << "\t" << time << "\t" << t1_2d_collapse / time << "\n";
    }

    // 2D Transpose with Collapse
    cout << "\n2D Transpose with Collapse:\n";
    cout << "Threads\tTime(s)\tSpeedup\n";
    double t1_2d_trans_collapse = 0.0;
    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
        fill(C.begin(), C.end(), 0.0);
        double start = omp_get_wtime();
        for (int run = 0; run < 1; run++)
        {
#pragma omp parallel for num_threads(threads) collapse(2)
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++)
                    {
                        sum += A[i * N + k] * B_trans[j * N + k];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
        double time = omp_get_wtime() - start;
        if (threads == 1)
            t1_2d_trans_collapse = time;
        cout << threads << "\t" << time << "\t" << t1_2d_trans_collapse / time << "\n";
    }

    return 0;
}