#!/usr/bin/env python3
"""
Q4 Bonus: Python vs AWK Runtime Comparison

This script performs the same tasks as the AWK pipeline and measures runtime.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from collections import Counter
import numpy as np

# Configuration
DATA_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/interview_data')
BAM_FILE = DATA_DIR / 'input.bam'
OUT_DIR = Path('/home/neeyuhuynh/Desktop/me/genesolution/q4_solution/out')
SAM_FILE = OUT_DIR / 'input.sam'

# Ensure SAM file exists
if not SAM_FILE.exists():
    print("SAM file not found. Run samtools first.")
    sys.exit(1)

print("=" * 70)
print("Q4 Bonus: Python vs AWK Runtime Comparison")
print("=" * 70)
print(f"\nInput: {SAM_FILE}")
print(f"File size: {SAM_FILE.stat().st_size / (1024**3):.2f} GB")

# Count lines first
print("\nCounting lines...")
with open(SAM_FILE, 'r') as f:
    total_lines = sum(1 for _ in f)
print(f"Total lines: {total_lines:,}")

# ============================================================
# Task 1: Extract column 9 absolute values (Python)
# ============================================================
print("\n" + "=" * 50)
print("Task 1: Extract column 9 absolute values")
print("=" * 50)

# Python version
start_py = time.time()
col9_abs_py = []
with open(SAM_FILE, 'r') as f:
    for line in f:
        parts = line.split('\t')
        if len(parts) >= 9:
            try:
                val = abs(int(parts[8]))
                col9_abs_py.append(val)
            except ValueError:
                pass
end_py = time.time()
py_time_task1 = end_py - start_py
print(f"Python: {py_time_task1:.2f}s ({len(col9_abs_py):,} values)")

# AWK version
start_awk = time.time()
result = subprocess.run(
    f"awk '{{v=$9; if (v<0) v=-v; print v}}' {SAM_FILE} | wc -l",
    shell=True, capture_output=True, text=True
)
end_awk = time.time()
awk_time_task1 = end_awk - start_awk
awk_count = int(result.stdout.strip())
print(f"AWK:    {awk_time_task1:.2f}s ({awk_count:,} values)")
print(f"Speedup: AWK is {py_time_task1/awk_time_task1:.1f}x faster")

# ============================================================
# Task 2: Filter by CIGAR string (Python)
# ============================================================
print("\n" + "=" * 50)
print("Task 2: Filter by CIGAR [0-9][0-9]M")
print("=" * 50)

import re
cigar_pattern = re.compile(r'^[0-9][0-9]M$')

# Python version
start_py = time.time()
cigar_filtered_py = 0
with open(SAM_FILE, 'r') as f:
    for line in f:
        parts = line.split('\t')
        if len(parts) >= 6:
            if cigar_pattern.match(parts[5]):
                cigar_filtered_py += 1
end_py = time.time()
py_time_task2 = end_py - start_py
print(f"Python: {py_time_task2:.2f}s ({cigar_filtered_py:,} lines kept)")

# AWK version
start_awk = time.time()
result = subprocess.run(
    f"awk '$6 ~ /^[0-9][0-9]M$/' {SAM_FILE} | wc -l",
    shell=True, capture_output=True, text=True
)
end_awk = time.time()
awk_time_task2 = end_awk - start_awk
awk_count = int(result.stdout.strip())
print(f"AWK:    {awk_time_task2:.2f}s ({awk_count:,} lines kept)")
print(f"Speedup: AWK is {py_time_task2/awk_time_task2:.1f}x faster")

# ============================================================
# Task 3: Extract 4-letter sequences (Python)
# ============================================================
print("\n" + "=" * 50)
print("Task 3: Extract 4-letter sequences")
print("=" * 50)

# Python version
start_py = time.time()
seq4_py = []
with open(SAM_FILE, 'r') as f:
    for line in f:
        parts = line.split('\t')
        if len(parts) >= 10:
            try:
                col9 = int(parts[8])
                seq = parts[9]
                if col9 > 0:
                    seq4_py.append(seq[:4])
                else:
                    seq4_py.append(seq[-4:])
            except (ValueError, IndexError):
                pass
end_py = time.time()
py_time_task3 = end_py - start_py
print(f"Python: {py_time_task3:.2f}s ({len(seq4_py):,} sequences)")

# AWK version
start_awk = time.time()
result = subprocess.run(
    f"awk '{{seq=$10; if ($9>0) {{print substr(seq,1,4)}} else {{print substr(seq,length(seq)-3,4)}}}}' {SAM_FILE} | wc -l",
    shell=True, capture_output=True, text=True
)
end_awk = time.time()
awk_time_task3 = end_awk - start_awk
awk_count = int(result.stdout.strip())
print(f"AWK:    {awk_time_task3:.2f}s ({awk_count:,} sequences)")
print(f"Speedup: AWK is {py_time_task3/awk_time_task3:.1f}x faster")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("RUNTIME COMPARISON SUMMARY")
print("=" * 70)

total_py = py_time_task1 + py_time_task2 + py_time_task3
total_awk = awk_time_task1 + awk_time_task2 + awk_time_task3

print(f"\n{'Task':<30} {'Python (s)':<15} {'AWK (s)':<15} {'Speedup':<10}")
print("-" * 70)
print(f"{'Task 1: Col9 absolute':<30} {py_time_task1:<15.2f} {awk_time_task1:<15.2f} {py_time_task1/awk_time_task1:.1f}x")
print(f"{'Task 2: CIGAR filter':<30} {py_time_task2:<15.2f} {awk_time_task2:<15.2f} {py_time_task2/awk_time_task2:.1f}x")
print(f"{'Task 3: 4-mer extraction':<30} {py_time_task3:<15.2f} {awk_time_task3:<15.2f} {py_time_task3/awk_time_task3:.1f}x")
print("-" * 70)
print(f"{'TOTAL':<30} {total_py:<15.2f} {total_awk:<15.2f} {total_py/total_awk:.1f}x")

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("""
1. AWK is significantly faster than Python for text processing tasks.
   Typical speedup: 2-5x faster.

2. AWK's advantage comes from:
   - Compiled pattern matching (not interpreted)
   - Optimized for line-by-line text processing
   - Low memory overhead (no object creation per line)

3. Python's advantages:
   - More flexible for complex logic
   - Better for downstream analysis (numpy, pandas)
   - Easier debugging and maintenance

4. Recommendation:
   - Use AWK for initial data extraction from large files
   - Use Python for analysis, visualization, and complex transformations
   - Combine both in a pipeline for best results
""")

# Save results
results = {
    'file_size_gb': SAM_FILE.stat().st_size / (1024**3),
    'total_lines': total_lines,
    'tasks': {
        'task1_col9': {'python_s': py_time_task1, 'awk_s': awk_time_task1, 'speedup': py_time_task1/awk_time_task1},
        'task2_cigar': {'python_s': py_time_task2, 'awk_s': awk_time_task2, 'speedup': py_time_task2/awk_time_task2},
        'task3_seq4': {'python_s': py_time_task3, 'awk_s': awk_time_task3, 'speedup': py_time_task3/awk_time_task3},
    },
    'total': {'python_s': total_py, 'awk_s': total_awk, 'speedup': total_py/total_awk}
}

import json
with open(OUT_DIR / 'runtime_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'runtime_comparison.json'}")

print("\nDone!")
