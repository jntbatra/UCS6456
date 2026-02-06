
# Assignment 1: Parallel & Distributed Computing

**Course:** Parallel & Distributed Computing
**Student:** Jayant Batra
**Instructor:** Dr. Saif Nalbad

---

## Question 1: OpenMP Parallelization of DAXPY

**Question**
Analyze the speedup obtained by parallelizing the DAXPY operation using OpenMP while increasing the number of threads. Identify the thread count that gives maximum speedup and explain the performance behavior beyond that point.

**Example**
The DAXPY loop ( X[i] = a \cdot X[i] + Y[i] ) was executed with varying thread counts on vectors of size (2^{16}). While speedup improved initially, the best observed speedup (~5×) occurred at 16 threads on Linux (WSL). Beyond this, efficiency dropped sharply due to memory bandwidth saturation and thread oversubscription, even though CPU utilization increased.

**What I Learned**
I learned that memory-bound workloads like DAXPY scale poorly beyond a small number of threads. Amdahl’s Law alone is insufficient to predict performance because memory bandwidth becomes the dominant bottleneck, and adding threads can reduce efficiency despite higher CPU usage.

---

## Question 2: Parallel Matrix Multiplication Using OpenMP

**Question**
Implement and compare different OpenMP parallelization strategies for large matrix multiplication (1000×1000), and analyze their performance scaling and microarchitectural behavior.

**Example**
Four versions were implemented: 1D parallelization, 1D with transposed matrix B, 2D parallelization using `collapse(2)`, and 2D with transposed B. The 2D + transpose version consistently achieved the lowest execution time and best scalability, reaching over 8× speedup at higher thread counts due to improved cache locality and load balancing.

**What I Learned**
I learned that compute-intensive workloads benefit significantly from data reuse and cache-friendly memory access. Loop collapsing and matrix transposition can dramatically improve performance, and matrix multiplication scales much better than memory-bound kernels before hitting hardware limits.

---

## Question 3: Parallel Computation of π Using OpenMP

**Question**
Parallelize the numerical computation of π using OpenMP and analyze how performance scales with increasing thread count.

**Example**
π was computed using numerical integration with 100 million iterations and OpenMP reduction. The program showed near-linear speedup up to 8 threads (~4.7×). Beyond this point, execution time improvements were marginal, while total CPU work increased due to synchronization overhead, frequency throttling, and reduced IPC.

**What I Learned**
I learned that even compute-bound workloads have a practical scalability limit. Hardware constraints such as SMT contention, reduction overhead, and frequency scaling eventually dominate, making additional threads inefficient beyond the optimal range.

---

## Overall Takeaway

This assignment demonstrated that **parallel performance depends strongly on workload characteristics**. Memory-bound tasks scale early and saturate quickly, while compute-heavy tasks scale further but still hit hardware limits. Effective parallel programming requires understanding both algorithms and the underlying architecture, not just increasing thread count.
