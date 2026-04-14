/*
 * Question 1: Molecular Dynamics - Lennard-Jones Force Calculation
 * UCS645: Parallel & Distributed Computing | Assignment 2
 *
 * Computes Lennard-Jones potential forces for N particles in 3D space.
 * Uses OpenMP for parallelization with dynamic scheduling, atomic updates,
 * and reduction for total energy.
 *
 * Compile: g++ -O3 -fopenmp q1.cpp -o q1 -lm
 * Run:     ./q1 <num_particles> <max_threads>
 * Example: ./q1 1000 8
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <iomanip>

// Lennard-Jones parameters
const double EPSILON = 1.0;
const double SIGMA   = 1.0;
const double CUTOFF  = 2.5 * SIGMA;
const double CUTOFF2 = CUTOFF * CUTOFF;

// Pre-compute LJ constants
const double SIGMA6  = SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA;
const double SIGMA12 = SIGMA6 * SIGMA6;

struct Particle {
    double x, y, z;      // position
    double fx, fy, fz;   // force
};

// Initialize particles randomly in a cubic box
void initialize_particles(Particle* particles, int N, double box_size) {
    srand(42); // fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        particles[i].x  = ((double)rand() / RAND_MAX) * box_size;
        particles[i].y  = ((double)rand() / RAND_MAX) * box_size;
        particles[i].z  = ((double)rand() / RAND_MAX) * box_size;
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }
}

// Compute LJ force between two particles
inline void compute_LJ_force(const Particle& pi, const Particle& pj,
                              double& fx, double& fy, double& fz, double& energy) {
    double dx = pj.x - pi.x;
    double dy = pj.y - pi.y;
    double dz = pj.z - pi.z;

    double r2 = dx * dx + dy * dy + dz * dz;

    if (r2 < CUTOFF2 && r2 > 1e-12) {
        double r2_inv  = 1.0 / r2;
        double r6_inv  = r2_inv * r2_inv * r2_inv;
        double sr6     = SIGMA6 * r6_inv;
        double sr12    = sr6 * sr6;

        // Force magnitude / r
        double f_over_r = 24.0 * EPSILON * (2.0 * sr12 - sr6) * r2_inv;

        fx = f_over_r * dx;
        fy = f_over_r * dy;
        fz = f_over_r * dz;

        energy = 4.0 * EPSILON * (sr12 - sr6);
    } else {
        fx = fy = fz = energy = 0.0;
    }
}

// Serial force computation
double compute_forces_serial(Particle* particles, int N) {
    // Reset forces
    for (int i = 0; i < N; i++) {
        particles[i].fx = particles[i].fy = particles[i].fz = 0.0;
    }

    double total_energy = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double fx, fy, fz, energy;
            compute_LJ_force(particles[i], particles[j], fx, fy, fz, energy);

            particles[i].fx += fx;
            particles[i].fy += fy;
            particles[i].fz += fz;

            // Newton's 3rd law
            particles[j].fx -= fx;
            particles[j].fy -= fy;
            particles[j].fz -= fz;

            total_energy += energy;
        }
    }
    return total_energy;
}

// Parallel force computation using OpenMP
double compute_forces_parallel(Particle* particles, int N, int num_threads) {
    // Reset forces
    for (int i = 0; i < N; i++) {
        particles[i].fx = particles[i].fy = particles[i].fz = 0.0;
    }

    double total_energy = 0.0;

    omp_set_num_threads(num_threads);

    #pragma omp parallel reduction(+:total_energy)
    {
        // Thread-local force arrays to avoid atomic operations
        double* local_fx = new double[N]();
        double* local_fy = new double[N]();
        double* local_fz = new double[N]();

        #pragma omp for schedule(dynamic, 10) nowait
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double fx, fy, fz, energy;
                compute_LJ_force(particles[i], particles[j], fx, fy, fz, energy);

                local_fx[i] += fx;
                local_fy[i] += fy;
                local_fz[i] += fz;

                // Newton's 3rd law
                local_fx[j] -= fx;
                local_fy[j] -= fy;
                local_fz[j] -= fz;

                total_energy += energy;
            }
        }

        // Accumulate thread-local forces into global arrays
        #pragma omp critical
        {
            for (int i = 0; i < N; i++) {
                particles[i].fx += local_fx[i];
                particles[i].fy += local_fy[i];
                particles[i].fz += local_fz[i];
            }
        }

        delete[] local_fx;
        delete[] local_fy;
        delete[] local_fz;
    }

    return total_energy;
}

int main(int argc, char* argv[]) {
    int N = 1000;           // default particle count
    int max_threads = 8;    // default max threads

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) max_threads = atoi(argv[2]);

    double box_size = 10.0 * pow((double)N / 1000.0, 1.0 / 3.0);

    Particle* particles = new Particle[N];

    std::cout << "============================================================\n";
    std::cout << "  Molecular Dynamics: Lennard-Jones Force Calculation\n";
    std::cout << "  Particles: " << N << "  |  Max Threads: " << max_threads << "\n";
    std::cout << "  Box Size: " << std::fixed << std::setprecision(2) << box_size << "\n";
    std::cout << "  Cutoff: " << CUTOFF << " sigma\n";
    std::cout << "============================================================\n\n";

    // Header
    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(16) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Efficiency %"
              << std::setw(22) << "Energy"
              << "\n";
    std::cout << std::string(76, '-') << "\n";

    double serial_time = 0.0;
    double serial_energy = 0.0;

    // Test with different thread counts: 1, 2, 4, 8, ...
    for (int t = 1; t <= max_threads; t *= 2) {
        // Re-initialize particles for each run
        initialize_particles(particles, N, box_size);

        double start = omp_get_wtime();
        double energy;
        if (t == 1) {
            energy = compute_forces_serial(particles, N);
        } else {
            energy = compute_forces_parallel(particles, N, t);
        }
        double end = omp_get_wtime();
        double elapsed = end - start;

        if (t == 1) {
            serial_time   = elapsed;
            serial_energy = energy;
        }

        double speedup    = serial_time / elapsed;
        double efficiency = (speedup / t) * 100.0;

        std::cout << std::left
                  << std::setw(10) << t
                  << std::setw(16) << std::fixed << std::setprecision(6) << elapsed
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(16) << std::fixed << std::setprecision(2) << efficiency
                  << std::setw(22) << std::fixed << std::setprecision(0) << energy
                  << "\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Energy Conservation Check:\n";
    std::cout << "  Serial Energy:  " << std::fixed << std::setprecision(6) << serial_energy << "\n";
    std::cout << "  (All runs should show nearly identical energy values)\n";
    std::cout << "============================================================\n";

    delete[] particles;
    return 0;
}
