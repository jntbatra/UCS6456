#!/usr/bin/env python3
"""Generate performance graphs for LAB6 CUDA assignment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = 'graphs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

# === Part B: Array Sum ===
b_N      = [1024, 65536, 1048576, 16777216]
b_labels = ['1K', '64K', '1M', '16M']
b_cpu_ms = [0.001, 0.035, 0.567, 9.649]
b_gpu_ms = [0.071, 0.014, 2.699, 2.799]
b_speedup = [c/g for c,g in zip(b_cpu_ms, b_gpu_ms)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Part B: CUDA Array Sum — CPU vs GPU', fontweight='bold')
x = np.arange(len(b_labels)); w = 0.35
ax1.bar(x - w/2, b_cpu_ms, w, label='CPU', color='#e74c3c')
ax1.bar(x + w/2, b_gpu_ms, w, label='GPU', color='#3498db')
ax1.set_xlabel('Array Size (N)'); ax1.set_ylabel('Time (ms)'); ax1.set_title('Execution Time')
ax1.set_xticks(x); ax1.set_xticklabels(b_labels); ax1.legend(); ax1.set_yscale('log')
ax2.bar(range(len(b_labels)), b_speedup, color=['#e74c3c' if s<1 else '#2ecc71' for s in b_speedup], tick_label=b_labels)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
ax2.set_xlabel('Array Size (N)'); ax2.set_ylabel('Speedup (CPU/GPU)'); ax2.set_title('GPU Speedup')
ax2.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/b_array_sum.png', dpi=150); plt.close()

# === Part C: Matrix Addition ===
c_sizes   = ['512x512', '1024x1024', '2048x2048', '4096x4096']
c_cpu_ms  = [0.325, 2.535, 6.614, 22.850]
c_gpu_ms  = [3.219, 0.099, 0.205, 1.896]
c_speedup = [c/g for c,g in zip(c_cpu_ms, c_gpu_ms)]
c_bw      = [1.0, 127.4, 245.1, 106.2]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Part C: CUDA Matrix Addition Performance', fontweight='bold')

x = np.arange(len(c_sizes)); w = 0.35
axes[0].bar(x - w/2, c_cpu_ms, w, label='CPU', color='#e74c3c')
axes[0].bar(x + w/2, c_gpu_ms, w, label='GPU', color='#3498db')
axes[0].set_xlabel('Matrix Size'); axes[0].set_ylabel('Time (ms)'); axes[0].set_title('Execution Time')
axes[0].set_xticks(x); axes[0].set_xticklabels(c_sizes, rotation=20); axes[0].legend()

axes[1].plot(c_sizes, c_speedup, 'o-', color='#2ecc71', linewidth=2, markersize=8)
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Matrix Size'); axes[1].set_ylabel('Speedup'); axes[1].set_title('GPU Speedup')

axes[2].bar(range(len(c_sizes)), c_bw, color='#8e44ad', tick_label=c_sizes)
axes[2].axhline(y=384.0, color='red', linestyle='--', alpha=0.5, label='Theoretical BW (384 GB/s)')
axes[2].set_xlabel('Matrix Size'); axes[2].set_ylabel('GB/s'); axes[2].set_title('Effective Bandwidth')
axes[2].legend(fontsize=9)
for label in axes[2].get_xticklabels(): label.set_rotation(20)
plt.tight_layout(); plt.savefig(f'{OUT}/c_matrix_add.png', dpi=150); plt.close()

# === Dashboard ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Assignment 6: CUDA Introduction — Performance Dashboard', fontsize=16, fontweight='bold')

ax = axes[0,0]
x = np.arange(len(b_labels)); w = 0.35
ax.bar(x - w/2, b_cpu_ms, w, label='CPU', color='#e74c3c')
ax.bar(x + w/2, b_gpu_ms, w, label='GPU', color='#3498db')
ax.set_title('Array Sum: Time'); ax.set_xlabel('N'); ax.set_xticks(x); ax.set_xticklabels(b_labels)
ax.set_ylabel('Time (ms)'); ax.set_yscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0,1]
ax.bar(range(len(b_labels)), b_speedup, color=['#e74c3c' if s<1 else '#2ecc71' for s in b_speedup], tick_label=b_labels)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Array Sum: Speedup'); ax.set_xlabel('N'); ax.set_ylabel('Speedup'); ax.grid(True, alpha=0.3)

ax = axes[1,0]
x = np.arange(len(c_sizes)); w = 0.35
ax.bar(x - w/2, c_cpu_ms, w, label='CPU', color='#e74c3c')
ax.bar(x + w/2, c_gpu_ms, w, label='GPU', color='#3498db')
ax.set_title('Matrix Add: Time'); ax.set_xlabel('Size'); ax.set_xticks(x); ax.set_xticklabels(c_sizes, rotation=15)
ax.set_ylabel('Time (ms)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.bar(range(len(c_sizes)), c_bw, color='#8e44ad', tick_label=c_sizes)
ax.axhline(y=384.0, color='red', linestyle='--', alpha=0.5, label='Peak BW')
ax.set_title('Matrix Add: Bandwidth'); ax.set_xlabel('Size'); ax.set_ylabel('GB/s')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels(): label.set_rotation(15)

plt.tight_layout(); plt.savefig(f'{OUT}/dashboard.png', dpi=150); plt.close()

print("All graphs generated in graphs/")
