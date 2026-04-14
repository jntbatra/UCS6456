/*
 * Question 2: Bioinformatics - DNA Sequence Alignment (Smith-Waterman)
 * UCS645: Parallel & Distributed Computing | Assignment 2
 *
 * Parallel Smith-Waterman local sequence alignment using OpenMP.
 * Implements wavefront (anti-diagonal) parallelization to handle
 * data dependencies in the DP matrix.
 *
 * Compile: g++ -O3 -fopenmp q2.cpp -o q2 -lm
 * Run:     ./q2 <seq_length> <max_threads>
 * Example: ./q2 2000 8
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <string>
#include <vector>

// Scoring parameters
const int MATCH    =  2;
const int MISMATCH = -1;
const int GAP      = -2;

// Generate a random DNA sequence of given length
std::string generate_dna_sequence(int length, unsigned int seed) {
    const char bases[] = "ACGT";
    std::string seq(length, 'A');
    srand(seed);
    for (int i = 0; i < length; i++) {
        seq[i] = bases[rand() % 4];
    }
    return seq;
}

// Scoring function
inline int score(char a, char b) {
    return (a == b) ? MATCH : MISMATCH;
}

// ========================
// Serial Smith-Waterman
// ========================
int smith_waterman_serial(const std::string& seq1, const std::string& seq2,
                          int& best_i, int& best_j) {
    int m = seq1.length();
    int n = seq2.length();

    // Allocate scoring matrix
    int** H = new int*[m + 1];
    for (int i = 0; i <= m; i++) {
        H[i] = new int[n + 1]();
    }

    int max_score = 0;
    best_i = 0;
    best_j = 0;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
            int up   = H[i-1][j]   + GAP;
            int left = H[i][j-1]   + GAP;

            H[i][j] = std::max({0, diag, up, left});

            if (H[i][j] > max_score) {
                max_score = H[i][j];
                best_i = i;
                best_j = j;
            }
        }
    }

    for (int i = 0; i <= m; i++) delete[] H[i];
    delete[] H;

    return max_score;
}

// ============================================================
// Parallel Smith-Waterman with Wavefront (Anti-Diagonal) Method
// ============================================================
int smith_waterman_wavefront(const std::string& seq1, const std::string& seq2,
                             int num_threads, int& best_i, int& best_j) {
    int m = seq1.length();
    int n = seq2.length();

    // Allocate scoring matrix
    int** H = new int*[m + 1];
    for (int i = 0; i <= m; i++) {
        H[i] = new int[n + 1]();
    }

    int max_score = 0;
    best_i = 0;
    best_j = 0;

    omp_set_num_threads(num_threads);

    // Wavefront: process anti-diagonals
    // Anti-diagonal d goes from 2 to (m + n)
    // On anti-diagonal d: i + j = d  =>  j = d - i
    int total_diags = m + n;

    for (int d = 2; d <= total_diags; d++) {
        int i_start = std::max(1, d - n);
        int i_end   = std::min(m, d - 1);
        int diag_len = i_end - i_start + 1;

        int local_max = 0;
        int local_bi = 0, local_bj = 0;

        #pragma omp parallel for schedule(static) reduction(max:local_max) \
            if(diag_len > 64)
        for (int i = i_start; i <= i_end; i++) {
            int j = d - i;

            int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
            int up   = H[i-1][j]   + GAP;
            int left = H[i][j-1]   + GAP;

            H[i][j] = std::max({0, diag, up, left});

            if (H[i][j] > local_max) {
                local_max = H[i][j];
            }
        }

        if (local_max > max_score) {
            // Find exact position (serial, rare)
            for (int i = i_start; i <= i_end; i++) {
                int j = d - i;
                if (H[i][j] == local_max && local_max > max_score) {
                    max_score = local_max;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }

    for (int i = 0; i <= m; i++) delete[] H[i];
    delete[] H;

    return max_score;
}

// ============================================================
// Parallel Smith-Waterman with Row-based parallelism
// (Uses different scheduling strategies for comparison)
// ============================================================
int smith_waterman_row_parallel(const std::string& seq1, const std::string& seq2,
                                int num_threads, const std::string& sched_type,
                                int& best_i, int& best_j) {
    int m = seq1.length();
    int n = seq2.length();

    // Allocate scoring matrix
    int** H = new int*[m + 1];
    for (int i = 0; i <= m; i++) {
        H[i] = new int[n + 1]();
    }

    int max_score = 0;
    best_i = 0;
    best_j = 0;

    omp_set_num_threads(num_threads);

    // Process row by row, parallelize within each row where safe
    // Anti-diagonal wavefront for correctness
    int total_diags = m + n;
    for (int d = 2; d <= total_diags; d++) {
        int i_start = std::max(1, d - n);
        int i_end   = std::min(m, d - 1);

        int local_max = 0;

        if (sched_type == "static") {
            #pragma omp parallel for schedule(static) reduction(max:local_max)
            for (int i = i_start; i <= i_end; i++) {
                int j = d - i;
                int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int up   = H[i-1][j]   + GAP;
                int left = H[i][j-1]   + GAP;
                H[i][j]  = std::max({0, diag, up, left});
                if (H[i][j] > local_max) local_max = H[i][j];
            }
        } else if (sched_type == "dynamic") {
            #pragma omp parallel for schedule(dynamic, 32) reduction(max:local_max)
            for (int i = i_start; i <= i_end; i++) {
                int j = d - i;
                int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int up   = H[i-1][j]   + GAP;
                int left = H[i][j-1]   + GAP;
                H[i][j]  = std::max({0, diag, up, left});
                if (H[i][j] > local_max) local_max = H[i][j];
            }
        } else { // guided
            #pragma omp parallel for schedule(guided) reduction(max:local_max)
            for (int i = i_start; i <= i_end; i++) {
                int j = d - i;
                int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int up   = H[i-1][j]   + GAP;
                int left = H[i][j-1]   + GAP;
                H[i][j]  = std::max({0, diag, up, left});
                if (H[i][j] > local_max) local_max = H[i][j];
            }
        }

        if (local_max > max_score) {
            for (int i = i_start; i <= i_end; i++) {
                int j = d - i;
                if (H[i][j] == local_max && local_max > max_score) {
                    max_score = local_max;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }

    for (int i = 0; i <= m; i++) delete[] H[i];
    delete[] H;

    return max_score;
}

int main(int argc, char* argv[]) {
    int seq_len = 2000;
    int max_threads = 8;

    if (argc >= 2) seq_len = atoi(argv[1]);
    if (argc >= 3) max_threads = atoi(argv[2]);

    // Generate two random DNA sequences
    std::string seq1 = generate_dna_sequence(seq_len, 42);
    std::string seq2 = generate_dna_sequence(seq_len, 123);

    std::cout << "============================================================\n";
    std::cout << "  DNA Sequence Alignment: Smith-Waterman Algorithm\n";
    std::cout << "  Sequence 1 Length: " << seq1.length() << "\n";
    std::cout << "  Sequence 2 Length: " << seq2.length() << "\n";
    std::cout << "  Scoring: Match=" << MATCH << " Mismatch=" << MISMATCH
              << " Gap=" << GAP << "\n";
    std::cout << "  Max Threads: " << max_threads << "\n";
    std::cout << "============================================================\n";

    // ===== Part 1: Thread Scaling with Wavefront =====
    std::cout << "\n--- Part 1: Wavefront Parallelization (Thread Scaling) ---\n\n";
    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(16) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Efficiency %"
              << std::setw(12) << "Score"
              << std::setw(12) << "Best_i"
              << std::setw(12) << "Best_j"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    double serial_time = 0.0;
    int ref_score = 0;

    for (int t = 1; t <= max_threads; t *= 2) {
        int bi, bj;

        double start = omp_get_wtime();
        int max_score;
        if (t == 1) {
            max_score = smith_waterman_serial(seq1, seq2, bi, bj);
        } else {
            max_score = smith_waterman_wavefront(seq1, seq2, t, bi, bj);
        }
        double end = omp_get_wtime();
        double elapsed = end - start;

        if (t == 1) {
            serial_time = elapsed;
            ref_score = max_score;
        }

        double speedup    = serial_time / elapsed;
        double efficiency = (speedup / t) * 100.0;

        std::cout << std::left
                  << std::setw(10) << t
                  << std::setw(16) << std::fixed << std::setprecision(6) << elapsed
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(16) << std::fixed << std::setprecision(2) << efficiency
                  << std::setw(12) << max_score
                  << std::setw(12) << bi
                  << std::setw(12) << bj
                  << "\n";
    }

    // ===== Part 2: Scheduling Strategy Comparison =====
    std::cout << "\n--- Part 2: Scheduling Strategy Comparison (threads=" << max_threads << ") ---\n\n";
    std::cout << std::left
              << std::setw(14) << "Schedule"
              << std::setw(16) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Efficiency %"
              << std::setw(12) << "Score"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::string schedules[] = {"static", "dynamic", "guided"};
    for (const auto& sched : schedules) {
        int bi, bj;
        double start = omp_get_wtime();
        int max_score = smith_waterman_row_parallel(seq1, seq2, max_threads, sched, bi, bj);
        double end = omp_get_wtime();
        double elapsed = end - start;

        double speedup    = serial_time / elapsed;
        double efficiency = (speedup / max_threads) * 100.0;

        std::cout << std::left
                  << std::setw(14) << sched
                  << std::setw(16) << std::fixed << std::setprecision(6) << elapsed
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(16) << std::fixed << std::setprecision(2) << efficiency
                  << std::setw(12) << max_score
                  << "\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Correctness Check:\n";
    std::cout << "  Reference Score (Serial): " << ref_score << "\n";
    std::cout << "  (All runs should produce identical alignment scores)\n";
    std::cout << "============================================================\n";

    return 0;
}
