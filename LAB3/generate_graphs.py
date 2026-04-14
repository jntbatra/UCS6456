#!/usr/bin/env python3
"""Generate performance analysis graphs for LAB3 correlation assignment."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = 'graphs'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============ DATA FROM BENCHMARK RUN ============

# Part 1: Size scaling
sizes = [100, 500, 1000, 2000]
t_seq   = [0.0019, 0.0449, 0.1763, 0.7265]
t_par   = [0.0002, 0.0041, 0.0140, 0.0534]
t_opt   = [0.0001, 0.0017, 0.0051, 0.0162]
sp_par  = [s/p for s,p in zip(t_seq, t_par)]
sp_opt  = [s/o for s,o in zip(t_seq, t_opt)]

# Part 2: Thread scaling (ny=1000, nx=1000)
threads     = [1,    2,      4,      8,      16,     20,     28]
t_par_th    = [0.1797, 0.0959, 0.0533, 0.0320, 0.0215, 0.0188, 0.0134]
t_opt_th    = [0.1797, 0.0255, 0.0134, 0.0085, 0.0055, 0.0055, 0.0053]
sp_par_th   = [0.1797/t for t in t_par_th]
sp_opt_th   = [0.1797/t for t in t_opt_th]
eff_par_th  = [s/t*100 for s,t in zip(sp_par_th, threads)]
eff_opt_th  = [s/t*100 for s,t in zip(sp_opt_th, threads)]


# ===== Graph 1: Execution Time vs Matrix Size =====
fig, ax = plt.subplots()
x = np.arange(len(sizes))
w = 0.25
ax.bar(x - w, t_seq, w, label='Sequential', color='#e74c3c')
ax.bar(x,     t_par, w, label='Parallel (OpenMP)', color='#3498db')
ax.bar(x + w, t_opt, w, label='Optimized (AVX2+OpenMP)', color='#2ecc71')
ax.set_xlabel('Matrix Rows (ny)')
ax.set_ylabel('Execution Time (s)')
ax.set_title('Execution Time vs Problem Size')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{OUT}/size_execution_time.png', dpi=150)
plt.close()


# ===== Graph 2: Speedup vs Matrix Size =====
fig, ax = plt.subplots()
ax.plot(sizes, sp_par, 'o-', label='Parallel (OpenMP)', color='#3498db', linewidth=2, markersize=8)
ax.plot(sizes, sp_opt, 's-', label='Optimized (AVX2+OpenMP)', color='#2ecc71', linewidth=2, markersize=8)
ax.set_xlabel('Matrix Rows (ny)')
ax.set_ylabel('Speedup (x)')
ax.set_title('Speedup vs Problem Size')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/size_speedup.png', dpi=150)
plt.close()


# ===== Graph 3: Thread Scaling — Execution Time =====
fig, ax = plt.subplots()
ax.plot(threads, t_par_th, 'o-', label='Parallel', color='#3498db', linewidth=2, markersize=8)
ax.plot(threads, t_opt_th, 's-', label='Optimized', color='#2ecc71', linewidth=2, markersize=8)
ax.set_xlabel('Number of Threads')
ax.set_ylabel('Execution Time (s)')
ax.set_title('Execution Time vs Thread Count (ny=1000, nx=1000)')
ax.legend()
ax.set_xticks(threads)
plt.tight_layout()
plt.savefig(f'{OUT}/thread_execution_time.png', dpi=150)
plt.close()


# ===== Graph 4: Thread Scaling — Speedup =====
fig, ax = plt.subplots()
ax.plot(threads, sp_par_th, 'o-', label='Parallel', color='#3498db', linewidth=2, markersize=8)
ax.plot(threads, sp_opt_th, 's-', label='Optimized', color='#2ecc71', linewidth=2, markersize=8)
ax.plot(threads, threads, '--', label='Ideal Linear', color='gray', alpha=0.5)
ax.set_xlabel('Number of Threads')
ax.set_ylabel('Speedup (x)')
ax.set_title('Speedup vs Thread Count')
ax.legend()
ax.set_xticks(threads)
plt.tight_layout()
plt.savefig(f'{OUT}/thread_speedup.png', dpi=150)
plt.close()


# ===== Graph 5: Parallel Efficiency =====
fig, ax = plt.subplots()
ax.plot(threads, eff_par_th, 'o-', label='Parallel', color='#3498db', linewidth=2, markersize=8)
ax.plot(threads, eff_opt_th, 's-', label='Optimized', color='#2ecc71', linewidth=2, markersize=8)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% (Ideal)')
ax.set_xlabel('Number of Threads')
ax.set_ylabel('Efficiency (%)')
ax.set_title('Parallel Efficiency vs Thread Count')
ax.legend()
ax.set_xticks(threads)
plt.tight_layout()
plt.savefig(f'{OUT}/thread_efficiency.png', dpi=150)
plt.close()


# ===== Graph 6: Dashboard =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Correlation Benchmark — Performance Dashboard', fontsize=16, fontweight='bold')

# Top-left: Size scaling bars
ax = axes[0, 0]
x = np.arange(len(sizes))
w = 0.25
ax.bar(x - w, t_seq, w, label='Sequential', color='#e74c3c')
ax.bar(x,     t_par, w, label='Parallel', color='#3498db')
ax.bar(x + w, t_opt, w, label='Optimized', color='#2ecc71')
ax.set_xlabel('ny'); ax.set_ylabel('Time (s)')
ax.set_title('Execution Time vs Size')
ax.set_xticks(x); ax.set_xticklabels(sizes)
ax.set_yscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Top-right: Thread speedup
ax = axes[0, 1]
ax.plot(threads, sp_par_th, 'o-', label='Parallel', color='#3498db', linewidth=2)
ax.plot(threads, sp_opt_th, 's-', label='Optimized', color='#2ecc71', linewidth=2)
ax.plot(threads, threads, '--', color='gray', alpha=0.5, label='Ideal')
ax.set_xlabel('Threads'); ax.set_ylabel('Speedup (x)')
ax.set_title('Speedup vs Threads')
ax.set_xticks(threads); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Bottom-left: Efficiency
ax = axes[1, 0]
ax.plot(threads, eff_par_th, 'o-', label='Parallel', color='#3498db', linewidth=2)
ax.plot(threads, eff_opt_th, 's-', label='Optimized', color='#2ecc71', linewidth=2)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Threads'); ax.set_ylabel('Efficiency (%)')
ax.set_title('Parallel Efficiency')
ax.set_xticks(threads); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Bottom-right: Size speedup
ax = axes[1, 1]
ax.plot(sizes, sp_par, 'o-', label='Parallel', color='#3498db', linewidth=2)
ax.plot(sizes, sp_opt, 's-', label='Optimized', color='#2ecc71', linewidth=2)
ax.set_xlabel('ny'); ax.set_ylabel('Speedup (x)')
ax.set_title('Speedup vs Problem Size')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/dashboard.png', dpi=150)
plt.close()

print("All graphs generated in graphs/")
