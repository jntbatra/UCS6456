#!/usr/bin/env python3
"""Generate performance graphs for LAB7 CUDA Part II assignment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = 'graphs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

# === Q1: Thread Tasks — Iterative vs Formula vs Parallel ===
q1_N      = [256, 1024, 4096, 16384, 65536]
q1_labels = ['256', '1K', '4K', '16K', '64K']
q1_iter   = [0.0075, 0.0152, 0.0474, 0.1796, 0.7094]
q1_form   = [0.0048, 0.0040, 0.0045, 0.0054, 0.0028]
q1_par    = [0.0053, 0.0052, 0.0045, 0.0045, 0.0051]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q1: Integer Sum — Iterative vs Formula vs Parallel Reduction', fontweight='bold')
x = np.arange(len(q1_labels)); w = 0.25
ax1.bar(x - w, q1_iter, w, label='Iterative (1 thread)', color='#e74c3c')
ax1.bar(x,     q1_form, w, label='Formula (O(1))',       color='#2ecc71')
ax1.bar(x + w, q1_par,  w, label='Parallel Reduction',   color='#3498db')
ax1.set_xlabel('N'); ax1.set_ylabel('Time (ms)'); ax1.set_title('Execution Time')
ax1.set_xticks(x); ax1.set_xticklabels(q1_labels); ax1.legend(fontsize=9); ax1.set_yscale('log')

speedup_par = [i/p for i,p in zip(q1_iter, q1_par)]
speedup_form = [i/f for i,f in zip(q1_iter, q1_form)]
ax2.plot(q1_labels, speedup_par, 'o-', color='#3498db', linewidth=2, markersize=8, label='Parallel vs Iter')
ax2.plot(q1_labels, speedup_form, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Formula vs Iter')
ax2.set_xlabel('N'); ax2.set_ylabel('Speedup over Iterative'); ax2.set_title('Speedup')
ax2.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/q1_thread_tasks.png', dpi=150); plt.close()

# === Q2: Merge Sort ===
q2_N     = [100, 500, 1000, 5000, 10000, 50000]
q2_labels = ['100', '500', '1K', '5K', '10K', '50K']
q2_cpu   = [0.0031, 0.0187, 0.0372, 0.2172, 0.4728, 2.7473]
q2_gpu   = [0.0314, 0.1047, 0.1946, 1.1158, 4.7590, 19.5917]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q2: Merge Sort — CPU Pipelined vs GPU Parallel', fontweight='bold')
x = np.arange(len(q2_labels)); w = 0.35
ax1.bar(x - w/2, q2_cpu, w, label='CPU', color='#e74c3c')
ax1.bar(x + w/2, q2_gpu, w, label='GPU', color='#3498db')
ax1.set_xlabel('Array Size'); ax1.set_ylabel('Time (ms)'); ax1.set_title('Execution Time')
ax1.set_xticks(x); ax1.set_xticklabels(q2_labels); ax1.legend(); ax1.set_yscale('log')

ratio = [c/g for c,g in zip(q2_cpu, q2_gpu)]
ax2.plot(q2_labels, ratio, 'o-', color='#8e44ad', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
ax2.set_xlabel('Array Size'); ax2.set_ylabel('CPU/GPU Ratio'); ax2.set_title('Speedup (>1 = GPU wins)')
ax2.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/q2_merge_sort.png', dpi=150); plt.close()

# === Q3: Vector Add Profiling — Bandwidth ===
q3_N     = [16384, 65536, 262144, 1048576, 4194304, 16777216]
q3_labels = ['16K', '64K', '256K', '1M', '4M', '16M']
q3_bw    = [35.9, 148.9, 603.1, 918.7, 385.1, 352.8]
q3_time  = [0.0055, 0.0053, 0.0052, 0.0137, 0.1307, 0.5706]
theoretical_bw = 384.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q3: vectorAdd Bandwidth vs Vector Size', fontweight='bold')

ax1.plot(q3_labels, q3_bw, 'o-', color='#3498db', linewidth=2, markersize=8, label='Measured BW')
ax1.axhline(y=theoretical_bw, color='red', linestyle='--', alpha=0.7, label=f'Theoretical ({theoretical_bw} GB/s)')
ax1.set_xlabel('Vector Size (N)'); ax1.set_ylabel('Bandwidth (GB/s)'); ax1.set_title('Effective Bandwidth')
ax1.legend(fontsize=9)

ax2.plot(q3_labels, q3_time, 's-', color='#2ecc71', linewidth=2, markersize=8)
ax2.set_xlabel('Vector Size (N)'); ax2.set_ylabel('Time (ms)'); ax2.set_title('Kernel Execution Time')
ax2.set_yscale('log')
plt.tight_layout(); plt.savefig(f'{OUT}/q3_vector_profiling.png', dpi=150); plt.close()

# === Dashboard ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Assignment 7: CUDA Part II — Performance Dashboard', fontsize=16, fontweight='bold')

ax = axes[0,0]
x = np.arange(len(q1_labels)); w = 0.25
ax.bar(x - w, q1_iter, w, label='Iterative', color='#e74c3c')
ax.bar(x,     q1_form, w, label='Formula', color='#2ecc71')
ax.bar(x + w, q1_par,  w, label='Parallel', color='#3498db')
ax.set_title('Q1: Sum Methods'); ax.set_xlabel('N'); ax.set_xticks(x); ax.set_xticklabels(q1_labels)
ax.set_yscale('log'); ax.set_ylabel('Time (ms)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0,1]
x = np.arange(len(q2_labels)); w = 0.35
ax.bar(x - w/2, q2_cpu, w, label='CPU', color='#e74c3c')
ax.bar(x + w/2, q2_gpu, w, label='GPU', color='#3498db')
ax.set_title('Q2: Merge Sort'); ax.set_xlabel('N'); ax.set_xticks(x); ax.set_xticklabels(q2_labels)
ax.set_yscale('log'); ax.set_ylabel('Time (ms)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.plot(q3_labels, q3_bw, 'o-', color='#3498db', linewidth=2, markersize=8)
ax.axhline(y=theoretical_bw, color='red', linestyle='--', alpha=0.7, label='Theoretical')
ax.set_title('Q3: Measured Bandwidth'); ax.set_xlabel('N'); ax.set_ylabel('GB/s')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1,1]
bar_data = [0.0281, 2.1489]
bar_labels = ['Dynamic\n(cudaMalloc)', 'Static\n(__device__)']
colors = ['#2ecc71', '#e74c3c']
ax.bar(bar_labels, bar_data, color=colors)
ax.set_title('Q3: Static vs Dynamic Alloc'); ax.set_ylabel('Time (ms)'); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig(f'{OUT}/dashboard.png', dpi=150); plt.close()

print("All graphs generated in graphs/")
