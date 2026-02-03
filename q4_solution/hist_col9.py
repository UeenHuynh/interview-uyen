#!/usr/bin/env python3
import csv
import sys

try:
    import numpy as np
except Exception as exc:
    print("numpy is required for hist_col9.py", file=sys.stderr)
    raise


def load_values(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
    return np.array(vals, dtype=float)


def write_hist_csv(out_csv, counts, bins):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_start", "bin_end", "count"])
        for i, c in enumerate(counts):
            w.writerow([bins[i], bins[i + 1], int(c)])


def main():
    if len(sys.argv) < 3:
        print("Usage: hist_col9.py <col9_abs.txt> <out_csv> [out_png]", file=sys.stderr)
        sys.exit(2)

    in_path = sys.argv[1]
    out_csv = sys.argv[2]
    out_png = sys.argv[3] if len(sys.argv) > 3 else None

    vals = load_values(in_path)
    if vals.size == 0:
        print("No values found", file=sys.stderr)
        sys.exit(1)

    # 50 bins is a reasonable default; change if needed
    counts, bins = np.histogram(vals, bins=50)
    write_hist_csv(out_csv, counts, bins)

    if out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.hist(vals, bins=50, color="#4C78A8", edgecolor="black")
            plt.title("Histogram of |col9|")
            plt.xlabel("|col9|")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception as exc:
            print("matplotlib not available; skip PNG", file=sys.stderr)


if __name__ == "__main__":
    main()
