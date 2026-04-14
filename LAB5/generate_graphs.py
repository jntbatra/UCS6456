#!/usr/bin/env python3
"""Generate performance graphs for LAB5 MPI Part II."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = 'graphs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

# === Q1: Broadcast Race ===
q1_procs = [2, 4, 8, 16]
q1_my    = [0.007313, 0.021777, 0.053517, 0.172196]
q1_mpi   = [0.007235, 0.021973, 0.017795, 0.069625]
q1_ratio = [m/b for m,b in zip(q1_my, q1_mpi)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q1: Broadcast Race — MyBcast vs MPI_Bcast (80 MB)', fontweight='bold')
x = np.arange(len(q1_procs)); w = 0.35
ax1.bar(x - w/2, [t*1000 for t in q1_my], w, label='MyBcast (linear)', color='#e74c3c')
ax1.bar(x + w/2, [t*1000 for t in q1_mpi], w, label='MPI_Bcast (tree)', color='#2ecc71')
ax1.set_xlabel('Processes'); ax1.set_ylabel('Time (ms)'); ax1.set_title('Execution Time')
ax1.set_xticks(x); ax1.set_xticklabels(q1_procs); ax1.legend()
ax2.plot(q1_procs, q1_ratio, 's-', color='#8e44ad', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Processes'); ax2.set_ylabel('MyBcast / MPI_Bcast')
ax2.set_title('MPI_Bcast Advantage'); ax2.set_xticks(q1_procs)
plt.tight_layout(); plt.savefig(f'{OUT}/q1_broadcast.png', dpi=150); plt.close()

# === Q2: Blocking vs Non-Blocking ===
labels = ['Blocking', 'Non-Blocking']
times = [0.0896, 0.0329]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, times, color=['#e74c3c', '#2ecc71'], width=0.5)
ax.set_ylabel('Time (s)')
ax.set_title('Q2: Blocking vs Non-Blocking Communication\n(200 MB transfer + 500M compute iterations)')
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{t:.4f}s', ha='center', va='bottom', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/q2_blocking.png', dpi=150); plt.close()

# === Q3: Dot Product ===
q3_procs   = [1, 2, 4, 8]
q3_compute = [0.309290, 0.217107, 0.123007, 0.146332]
q3_comm    = [0.000004, 0.016639, 0.000739, 0.053339]
q3_total   = [0.309295, 0.217129, 0.123026, 0.146503]
q3_speedup = [q3_total[0]/t for t in q3_total]
q3_eff     = [s/p*100 for s,p in zip(q3_speedup, q3_procs)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Q3: Distributed Dot Product — Amdahl\'s Law Analysis (500M elements)', fontsize=14, fontweight='bold')

ax = axes[0,0]
x = np.arange(len(q3_procs)); w = 0.35
ax.bar(x - w/2, q3_compute, w, label='Compute', color='#3498db')
ax.bar(x + w/2, q3_comm, w, label='Communication', color='#e74c3c')
ax.set_xlabel('Processes'); ax.set_ylabel('Time (s)'); ax.set_title('Compute vs Communication Time')
ax.set_xticks(x); ax.set_xticklabels(q3_procs); ax.legend()

ax = axes[0,1]
ax.plot(q3_procs, q3_speedup, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Actual')
ax.plot(q3_procs, q3_procs, '--', color='gray', alpha=0.5, label='Ideal')
ax.set_xlabel('Processes'); ax.set_ylabel('Speedup'); ax.set_title('Speedup')
ax.set_xticks(q3_procs); ax.legend()

ax = axes[1,0]
ax.plot(q3_procs, q3_eff, 'D-', color='#e74c3c', linewidth=2, markersize=8)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Processes'); ax.set_ylabel('Efficiency (%)'); ax.set_title('Parallel Efficiency')
ax.set_xticks(q3_procs)

ax = axes[1,1]
comm_pct = [c/t*100 for c,t in zip(q3_comm, q3_total)]
ax.bar(range(len(q3_procs)), comm_pct, color='#f39c12', tick_label=[str(p) for p in q3_procs])
ax.set_xlabel('Processes'); ax.set_ylabel('Comm Overhead (%)'); ax.set_title('Communication Overhead')

plt.tight_layout(); plt.savefig(f'{OUT}/q3_dotproduct.png', dpi=150); plt.close()

# === Dashboard ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Assignment 5: MPI Part II — Performance Dashboard', fontsize=16, fontweight='bold')

ax = axes[0,0]
x = np.arange(len(q1_procs)); w = 0.35
ax.bar(x-w/2, [t*1000 for t in q1_my], w, label='MyBcast', color='#e74c3c')
ax.bar(x+w/2, [t*1000 for t in q1_mpi], w, label='MPI_Bcast', color='#2ecc71')
ax.set_title('Q1: Broadcast Race (ms)'); ax.set_xticks(x); ax.set_xticklabels(q1_procs)
ax.set_xlabel('Processes'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0,1]
bars = ax.bar(labels, times, color=['#e74c3c', '#2ecc71'], width=0.5)
ax.set_title('Q2: Blocking vs Non-Blocking (s)'); ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.plot(q3_procs, q3_speedup, 's-', color='#2ecc71', linewidth=2, markersize=8)
ax.plot(q3_procs, q3_procs, '--', color='gray', alpha=0.5)
ax.set_title('Q3: Dot Product Speedup'); ax.set_xlabel('Processes')
ax.set_xticks(q3_procs); ax.grid(True, alpha=0.3)

ax = axes[1,1]
comm_pct = [c/t*100 for c,t in zip(q3_comm, q3_total)]
ax.bar(range(len(q3_procs)), comm_pct, color='#f39c12', tick_label=[str(p) for p in q3_procs])
ax.set_title('Q3: Communication Overhead (%)'); ax.set_xlabel('Processes'); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig(f'{OUT}/dashboard.png', dpi=150); plt.close()

print("All graphs generated in graphs/")
