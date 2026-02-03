#!/usr/bin/env python3
import csv
import math
import sys
from collections import Counter


def load_seqs(path):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if len(s) == 4:
                seqs.append(s)
    return seqs


def write_counts_csv(out_csv, counts):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq4", "count"])
        for k, v in counts.most_common():
            w.writerow([k, v])


def shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p, 2)
    return h


def main():
    if len(sys.argv) < 4:
        print("Usage: hist_seq4.py <seq4.txt> <out_csv> <out_png> [out_diversity]", file=sys.stderr)
        sys.exit(2)

    in_path = sys.argv[1]
    out_csv = sys.argv[2]
    out_png = sys.argv[3]
    out_div = sys.argv[4] if len(sys.argv) > 4 else None

    seqs = load_seqs(in_path)
    counts = Counter(seqs)
    write_counts_csv(out_csv, counts)

    if out_div:
        h = shannon_entropy(counts)
        with open(out_div, "w", encoding="utf-8") as f:
            f.write(f"shannon_entropy_bits\t{h:.6f}\n")
            f.write(f"unique_sequences\t{len(counts)}\n")
            f.write(f"total_sequences\t{sum(counts.values())}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [k for k, _ in counts.most_common(50)]
        values = [counts[k] for k in labels]

        plt.figure(figsize=(10, 4))
        plt.bar(labels, values, color="#F58518")
        plt.title("Top 50 4-mer frequencies")
        plt.xlabel("4-mer")
        plt.ylabel("Count")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        print("matplotlib not available; skip PNG", file=sys.stderr)


if __name__ == "__main__":
    main()
