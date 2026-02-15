/*
 * Question 3: Scientific Computing - Heat Diffusion Simulation
 * UCS645: Parallel & Distributed Computing | Assignment 2
 *
 * Simulates 2D heat diffusion in a metal plate using the finite difference
 * method with OpenMP parallelization. Explores static, dynamic, and guided
 * scheduling strategies.
 *
 * Compile: g++ -O3 -fopenmp q3.cpp -o q3 -lm
 * Run:     ./q3 <grid_size> <max_threads> <time_steps>
 * Example: ./q3 500 8 1000
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <iomanip>
#include <string>

// Physical parameters
const double ALPHA     = 0.01;    // Thermal diffusivity
const double DX        = 0.01;    // Spatial step in x
const double DY        = 0.01;    // Spatial step in y
const double DT        = 0.0001;  // Time step (must satisfy stability: dt <= dx^2/(4*alpha))

// Boundary conditions (fixed temperatures)
const double T_TOP     = 100.0;
const double T_BOTTOM  = 0.0;
const double T_LEFT    = 0.0;
const double T_RIGHT   = 0.0;
const double T_INITIAL = 25.0;    // Initial interior temperature

// Allocate 2D grid
double** allocate_grid(int N) {
    double** grid = new double*[N];
    for (int i = 0; i < N; i++) {
        grid[i] = new double[N];
    }
    return grid;
}

// Free 2D grid
void free_grid(double** grid, int N) {
    for (int i = 0; i < N; i++) {
        delete[] grid[i];
    }
    delete[] grid;
}

// Initialize grid with boundary and initial conditions
void initialize_grid(double** T, int N) {
    // Set interior to initial temperature
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            T[i][j] = T_INITIAL;

    // Set boundaries
    for (int j = 0; j < N; j++) {
        T[0][j]     = T_TOP;     // Top row
        T[N-1][j]   = T_BOTTOM;  // Bottom row
    }
    for (int i = 0; i < N; i++) {
        T[i][0]     = T_LEFT;    // Left column
        T[i][N-1]   = T_RIGHT;   // Right column
    }
}

// Copy grid
void copy_grid(double** src, double** dst, int N) {
    for (int i = 0; i < N; i++)
        memcpy(dst[i], src[i], N * sizeof(double));
}

// Compute total heat in the grid (for verification)
double compute_total_heat(double** T, int N) {
    double total = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total += T[i][j];
    return total;
}

// ========================
// Serial Heat Diffusion
// ========================
double simulate_serial(double** T, double** T_new, int N, int steps) {
    double cx = ALPHA * DT / (DX * DX);
    double cy = ALPHA * DT / (DY * DY);

    for (int step = 0; step < steps; step++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                T_new[i][j] = T[i][j]
                    + cx * (T[i+1][j] - 2.0*T[i][j] + T[i-1][j])
                    + cy * (T[i][j+1] - 2.0*T[i][j] + T[i][j-1]);
            }
        }
        // Swap pointers
        double** tmp = T;
        // Copy T_new back to T (keeping boundaries)
        for (int i = 1; i < N - 1; i++)
            memcpy(T[i] + 1, T_new[i] + 1, (N - 2) * sizeof(double));
    }

    return compute_total_heat(T, N);
}

// ========================
// Parallel Heat Diffusion
// ========================
double simulate_parallel(double** T, double** T_new, int N, int steps,
                         int num_threads, const std::string& sched_type) {
    double cx = ALPHA * DT / (DX * DX);
    double cy = ALPHA * DT / (DY * DY);

    omp_set_num_threads(num_threads);

    for (int step = 0; step < steps; step++) {
        if (sched_type == "static") {
            #pragma omp parallel for schedule(static)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + cx * (T[i+1][j] - 2.0*T[i][j] + T[i-1][j])
                        + cy * (T[i][j+1] - 2.0*T[i][j] + T[i][j-1]);
                }
            }
        } else if (sched_type == "dynamic") {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + cx * (T[i+1][j] - 2.0*T[i][j] + T[i-1][j])
                        + cy * (T[i][j+1] - 2.0*T[i][j] + T[i][j-1]);
                }
            }
        } else { // guided
            #pragma omp parallel for schedule(guided)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + cx * (T[i+1][j] - 2.0*T[i][j] + T[i-1][j])
                        + cy * (T[i][j+1] - 2.0*T[i][j] + T[i][j-1]);
                }
            }
        }

        // Copy T_new back to T (interior only)
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; i++) {
            memcpy(T[i] + 1, T_new[i] + 1, (N - 2) * sizeof(double));
        }
    }

    // Compute total heat using reduction
    double total = 0.0;
    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total += T[i][j];

    return total;
}

int main(int argc, char* argv[]) {
    int N = 500;             // grid size (N x N)
    int max_threads = 8;
    int steps = 1000;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) max_threads = atoi(argv[2]);
    if (argc >= 4) steps = atoi(argv[3]);

    // Stability check
    double cx = ALPHA * DT / (DX * DX);
    double cy = ALPHA * DT / (DY * DY);
    if (cx + cy >= 0.5) {
        std::cerr << "WARNING: Stability condition not met! cx+cy = "
                  << (cx + cy) << " >= 0.5\n";
        std::cerr << "Reduce DT or increase DX/DY.\n";
    }

    std::cout << "============================================================\n";
    std::cout << "  Heat Diffusion Simulation (2D Finite Difference)\n";
    std::cout << "  Grid Size: " << N << "x" << N << "\n";
    std::cout << "  Time Steps: " << steps << "\n";
    std::cout << "  Alpha: " << ALPHA << "  dx: " << DX << "  dy: " << DY
              << "  dt: " << DT << "\n";
    std::cout << "  Stability (cx+cy): " << std::fixed << std::setprecision(4)
              << (cx + cy) << " (must be < 0.5)\n";
    std::cout << "  Boundary: Top=" << T_TOP << " Bottom=" << T_BOTTOM
              << " Left=" << T_LEFT << " Right=" << T_RIGHT << "\n";
    std::cout << "  Max Threads: " << max_threads << "\n";
    std::cout << "============================================================\n";

    // ===== Part 1: Thread Scaling (static schedule) =====
    std::cout << "\n--- Part 1: Thread Scaling (schedule=static) ---\n\n";
    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(16) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Efficiency %"
              << std::setw(20) << "Total Heat"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    double serial_time = 0.0;
    double ref_heat = 0.0;

    for (int t = 1; t <= max_threads; t *= 2) {
        double** T     = allocate_grid(N);
        double** T_new = allocate_grid(N);
        initialize_grid(T, N);
        initialize_grid(T_new, N);

        double start = omp_get_wtime();
        double total_heat;
        if (t == 1) {
            total_heat = simulate_serial(T, T_new, N, steps);
        } else {
            total_heat = simulate_parallel(T, T_new, N, steps, t, "static");
        }
        double end = omp_get_wtime();
        double elapsed = end - start;

        if (t == 1) {
            serial_time = elapsed;
            ref_heat = total_heat;
        }

        double speedup    = serial_time / elapsed;
        double efficiency = (speedup / t) * 100.0;

        std::cout << std::left
                  << std::setw(10) << t
                  << std::setw(16) << std::fixed << std::setprecision(6) << elapsed
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(16) << std::fixed << std::setprecision(2) << efficiency
                  << std::setw(20) << std::fixed << std::setprecision(4) << total_heat
                  << "\n";

        free_grid(T, N);
        free_grid(T_new, N);
    }

    // ===== Part 2: Scheduling Strategy Comparison =====
    std::cout << "\n--- Part 2: Scheduling Strategy Comparison (threads=" << max_threads << ") ---\n\n";
    std::cout << std::left
              << std::setw(14) << "Schedule"
              << std::setw(16) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Efficiency %"
              << std::setw(20) << "Total Heat"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    std::string schedules[] = {"static", "dynamic", "guided"};
    for (const auto& sched : schedules) {
        double** T     = allocate_grid(N);
        double** T_new = allocate_grid(N);
        initialize_grid(T, N);
        initialize_grid(T_new, N);

        double start = omp_get_wtime();
        double total_heat = simulate_parallel(T, T_new, N, steps, max_threads, sched);
        double end = omp_get_wtime();
        double elapsed = end - start;

        double speedup    = serial_time / elapsed;
        double efficiency = (speedup / max_threads) * 100.0;

        std::cout << std::left
                  << std::setw(14) << sched
                  << std::setw(16) << std::fixed << std::setprecision(6) << elapsed
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(16) << std::fixed << std::setprecision(2) << efficiency
                  << std::setw(20) << std::fixed << std::setprecision(4) << total_heat
                  << "\n";

        free_grid(T, N);
        free_grid(T_new, N);
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Correctness Check:\n";
    std::cout << "  Reference Total Heat (Serial): " << std::fixed << std::setprecision(4) << ref_heat << "\n";
    std::cout << "  (All runs should produce nearly identical total heat)\n";
    std::cout << "============================================================\n";

    return 0;
}
