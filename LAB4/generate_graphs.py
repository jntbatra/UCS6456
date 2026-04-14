#!/usr/bin/env python3
"""Generate performance graphs for LAB4 MPI assignment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = 'graphs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

# === Q1: DAXPY ===
q1_procs = [1, 2, 4, 8]
q1_seq   = [0.000023, 0.000023, 0.000021, 0.000041]
q1_par   = [0.000041, 0.000059, 0.000072, 0.000129]
q1_speedup = [s/p for s,p in zip(q1_seq, q1_par)]
q1_eff = [sp/p*100 for sp,p in zip(q1_speedup, q1_procs)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q1: DAXPY MPI Performance (Vector Size = 65536)', fontweight='bold')
ax1.bar(range(len(q1_procs)), [t*1e6 for t in q1_par], color='#3498db', tick_label=[str(p) for p in q1_procs])
ax1.set_xlabel('Processes'); ax1.set_ylabel('Time (us)'); ax1.set_title('Execution Time')
ax2.plot(q1_procs, q1_speedup, 'o-', color='#e74c3c', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Processes'); ax2.set_ylabel('Speedup'); ax2.set_title('Speedup vs Sequential')
ax2.set_xticks(q1_procs)
plt.tight_layout(); plt.savefig(f'{OUT}/q1_daxpy.png', dpi=150); plt.close()

# === Q2: Broadcast Race ===
q2_procs = [2, 4, 8, 16]
q2_my    = [0.010539, 0.030236, 0.070122, 0.235912]
q2_mpi   = [0.007957, 0.022451, 0.019440, 0.071429]
q2_ratio = [m/b for m,b in zip(q2_my, q2_mpi)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q2: Broadcast Race — MyBcast vs MPI_Bcast (80 MB)', fontweight='bold')
x = np.arange(len(q2_procs)); w = 0.35
ax1.bar(x - w/2, q2_my, w, label='MyBcast (linear)', color='#e74c3c')
ax1.bar(x + w/2, q2_mpi, w, label='MPI_Bcast (tree)', color='#2ecc71')
ax1.set_xlabel('Processes'); ax1.set_ylabel('Time (s)'); ax1.set_title('Execution Time')
ax1.set_xticks(x); ax1.set_xticklabels(q2_procs); ax1.legend()
ax2.plot(q2_procs, q2_ratio, 's-', color='#8e44ad', linewidth=2, markersize=8)
ax2.set_xlabel('Processes'); ax2.set_ylabel('MyBcast / MPI_Bcast')
ax2.set_title('MPI_Bcast Advantage Ratio'); ax2.set_xticks(q2_procs)
plt.tight_layout(); plt.savefig(f'{OUT}/q2_broadcast.png', dpi=150); plt.close()

# === Q3: Dot Product ===
q3_procs = [1, 2, 4, 8]
q3_time  = [0.358586, 0.203848, 0.138250, 0.110644]
q3_speedup = [q3_time[0]/t for t in q3_time]
q3_eff = [s/p*100 for s,p in zip(q3_speedup, q3_procs)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Q3: Distributed Dot Product (500M elements)', fontweight='bold')
axes[0].plot(q3_procs, q3_time, 'o-', color='#3498db', linewidth=2, markersize=8)
axes[0].set_xlabel('Processes'); axes[0].set_ylabel('Time (s)'); axes[0].set_title('Execution Time')
axes[0].set_xticks(q3_procs)
axes[1].plot(q3_procs, q3_speedup, 's-', color='#2ecc71', linewidth=2, markersize=8)
axes[1].plot(q3_procs, q3_procs, '--', color='gray', alpha=0.5, label='Ideal')
axes[1].set_xlabel('Processes'); axes[1].set_ylabel('Speedup'); axes[1].set_title('Speedup')
axes[1].set_xticks(q3_procs); axes[1].legend()
axes[2].plot(q3_procs, q3_eff, 'D-', color='#e74c3c', linewidth=2, markersize=8)
axes[2].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Processes'); axes[2].set_ylabel('Efficiency (%)'); axes[2].set_title('Parallel Efficiency')
axes[2].set_xticks(q3_procs)
plt.tight_layout(); plt.savefig(f'{OUT}/q3_dotproduct.png', dpi=150); plt.close()

# === Q4: Primes ===
q4_procs = [2, 4, 8]
q4_time  = [0.0459, 0.0281, 0.0319]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(q4_procs)), q4_time, color=['#3498db','#2ecc71','#e74c3c'], tick_label=[str(p) for p in q4_procs])
ax.set_xlabel('Processes'); ax.set_ylabel('Time (s)')
ax.set_title('Q4: Prime Finder — Master-Slave (range 2-100000)')
plt.tight_layout(); plt.savefig(f'{OUT}/q4_primes.png', dpi=150); plt.close()

# === Q5: Perfect Numbers ===
q5_procs = [2, 4, 8]
q5_time  = [0.0727, 0.0350, 0.0338]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(q5_procs)), q5_time, color=['#3498db','#2ecc71','#e74c3c'], tick_label=[str(p) for p in q5_procs])
ax.set_xlabel('Processes'); ax.set_ylabel('Time (s)')
ax.set_title('Q5: Perfect Number Finder — Master-Slave (range 2-100000)')
plt.tight_layout(); plt.savefig(f'{OUT}/q5_perfect.png', dpi=150); plt.close()

# === Dashboard ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Assignment 4: MPI Performance Dashboard', fontsize=16, fontweight='bold')

ax = axes[0,0]
ax.bar(range(len(q1_procs)), [t*1e6 for t in q1_par], color='#3498db', tick_label=[str(p) for p in q1_procs])
ax.set_title('Q1: DAXPY Time (us)'); ax.set_xlabel('Processes'); ax.grid(True, alpha=0.3)

ax = axes[0,1]
x = np.arange(len(q2_procs)); w = 0.35
ax.bar(x-w/2, q2_my, w, label='MyBcast', color='#e74c3c')
ax.bar(x+w/2, q2_mpi, w, label='MPI_Bcast', color='#2ecc71')
ax.set_title('Q2: Broadcast Race'); ax.set_xlabel('Processes'); ax.set_xticks(x); ax.set_xticklabels(q2_procs)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0,2]
ax.plot(q3_procs, q3_speedup, 's-', color='#2ecc71', linewidth=2, markersize=8)
ax.plot(q3_procs, q3_procs, '--', color='gray', alpha=0.5)
ax.set_title('Q3: Dot Product Speedup'); ax.set_xlabel('Processes'); ax.set_xticks(q3_procs); ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.plot(q3_procs, q3_eff, 'D-', color='#e74c3c', linewidth=2, markersize=8)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Q3: Parallel Efficiency'); ax.set_xlabel('Processes'); ax.set_ylabel('%'); ax.set_xticks(q3_procs); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.bar(range(len(q4_procs)), q4_time, color='#8e44ad', tick_label=[str(p) for p in q4_procs])
ax.set_title('Q4: Prime Finder Time'); ax.set_xlabel('Processes'); ax.grid(True, alpha=0.3)

ax = axes[1,2]
ax.bar(range(len(q5_procs)), q5_time, color='#f39c12', tick_label=[str(p) for p in q5_procs])
ax.set_title('Q5: Perfect Numbers Time'); ax.set_xlabel('Processes'); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig(f'{OUT}/dashboard.png', dpi=150); plt.close()

print("All graphs generated in graphs/")
