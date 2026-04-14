#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

// Declared in correlate.cpp
void correlate(int ny, int nx, const float* data, float* result, int version);

void verify_results(int ny, const float* result1, const float* result2, const char* label) {
    double max_err = 0.0;
    for (int i = 0; i < ny; i++) {
        for (int j = i; j < ny; j++) {
            double err = std::abs((double)result1[i + j * ny] - (double)result2[i + j * ny]);
            max_err = std::max(max_err, err);
        }
    }
    std::cout << "  Verification (" << label << "): max error = " << std::scientific << max_err;
    if (max_err < 1e-5) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

double benchmark(int ny, int nx, const float* data, float* result, int version, int repeats = 3) {
    // Warmup
    correlate(ny, nx, data, result, version);

    double best = 1e30;
    for (int r = 0; r < repeats; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        correlate(ny, nx, data, result, version);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        best = std::min(best, elapsed);
    }
    return best;
}

int main(int argc, char* argv[]) {
    // Default sizes
    std::vector<int> sizes = {100, 500, 1000, 2000};
    int nx = 1000; // columns (vector length)
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 20, 28};

    if (argc >= 2) {
        sizes.clear();
        sizes.push_back(std::atoi(argv[1]));
    }
    if (argc >= 3) {
        nx = std::atoi(argv[2]);
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "  Pairwise Correlation Coefficient — Performance Analysis" << std::endl;
    std::cout << "  CPU: Intel i7-14700HX (20 cores, 28 threads)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // === PART 1: Size scaling (fixed threads) ===
    std::cout << "\n--- Part 1: Size Scaling (all threads) ---\n" << std::endl;
    std::cout << std::setw(8) << "ny" << std::setw(8) << "nx"
              << std::setw(14) << "Sequential" << std::setw(14) << "Parallel"
              << std::setw(14) << "Optimized"
              << std::setw(10) << "Speedup1" << std::setw(10) << "Speedup2" << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (int ny : sizes) {
        std::vector<float> data(ny * nx);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : data) v = dist(rng);

        std::vector<float> result_seq(ny * ny, 0.0f);
        std::vector<float> result_par(ny * ny, 0.0f);
        std::vector<float> result_opt(ny * ny, 0.0f);

        double t_seq = benchmark(ny, nx, data.data(), result_seq.data(), 1);
        double t_par = benchmark(ny, nx, data.data(), result_par.data(), 2);
        double t_opt = benchmark(ny, nx, data.data(), result_opt.data(), 3);

        std::cout << std::setw(8) << ny << std::setw(8) << nx
                  << std::setw(12) << std::fixed << std::setprecision(4) << t_seq << "s"
                  << std::setw(12) << t_par << "s"
                  << std::setw(12) << t_opt << "s"
                  << std::setw(9) << std::setprecision(2) << t_seq / t_par << "x"
                  << std::setw(9) << t_seq / t_opt << "x" << std::endl;

        verify_results(ny, result_seq.data(), result_par.data(), "parallel vs seq");
        verify_results(ny, result_seq.data(), result_opt.data(), "optimized vs seq");
    }

    // === PART 2: Thread scaling (fixed size) ===
    int test_ny = 1000;
    std::cout << "\n--- Part 2: Thread Scaling (ny=" << test_ny << ", nx=" << nx << ") ---\n" << std::endl;
    std::cout << std::setw(10) << "Threads" << std::setw(14) << "Parallel"
              << std::setw(14) << "Optimized"
              << std::setw(12) << "Speedup_P" << std::setw(12) << "Speedup_O"
              << std::setw(12) << "Efficiency" << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    std::vector<float> data(test_ny * nx);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rng);

    // Get sequential baseline
    std::vector<float> result_seq(test_ny * test_ny, 0.0f);
    double t_seq_base = benchmark(test_ny, nx, data.data(), result_seq.data(), 1);
    std::cout << std::setw(10) << "1 (seq)" << std::setw(12) << std::fixed << std::setprecision(4) << t_seq_base << "s"
              << std::setw(14) << "-"
              << std::setw(12) << "1.00x" << std::setw(12) << "-"
              << std::setw(12) << "100.0%" << std::endl;

    for (int nthreads : thread_counts) {
        if (nthreads == 1) continue; // already shown as sequential
        omp_set_num_threads(nthreads);

        std::vector<float> result_par(test_ny * test_ny, 0.0f);
        std::vector<float> result_opt(test_ny * test_ny, 0.0f);

        double t_par = benchmark(test_ny, nx, data.data(), result_par.data(), 2);
        double t_opt = benchmark(test_ny, nx, data.data(), result_opt.data(), 3);

        double sp_par = t_seq_base / t_par;
        double sp_opt = t_seq_base / t_opt;
        double eff = (sp_opt / nthreads) * 100.0;

        std::cout << std::setw(10) << nthreads
                  << std::setw(12) << std::fixed << std::setprecision(4) << t_par << "s"
                  << std::setw(12) << t_opt << "s"
                  << std::setw(11) << std::setprecision(2) << sp_par << "x"
                  << std::setw(11) << sp_opt << "x"
                  << std::setw(10) << std::setprecision(1) << eff << "%" << std::endl;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Done. Use 'perf stat ./correlate_bench <ny> <nx>' for HW counters." << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
