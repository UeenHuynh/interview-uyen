#!/usr/bin/env bash
set -euo pipefail

BAM_PATH="${1:-/home/neeyuhuynh/Desktop/me/genesolution/interview_data/input.bam}"
OUT_DIR="${2:-/home/neeyuhuynh/Desktop/me/genesolution/q4_solution/out}"

mkdir -p "$OUT_DIR"

# Convert BAM -> SAM text
SAM_TXT="$OUT_DIR/input.sam"

if ! command -v samtools >/dev/null 2>&1; then
  echo "samtools not found in PATH. Please install it and re-run." >&2
  exit 127
fi

# Rebuild if missing or empty
if [ ! -s "$SAM_TXT" ]; then
  samtools view "$BAM_PATH" > "$SAM_TXT"
fi

# 1) Column 9 absolute values
awk '{v=$9; if (v<0) v=-v; print v}' "$SAM_TXT" > "$OUT_DIR/col9_abs.txt"

# 2) Filter by CIGAR string: exactly two digits followed by M
awk '$6 ~ /^[0-9][0-9]M$/' "$SAM_TXT" > "$OUT_DIR/cigar_2digitM.sam"

# 3) Extract 4-letter sequences based on sign of column 9
awk '{seq=$10; if ($9>0) {print substr(seq,1,4)} else {print substr(seq,length(seq)-3,4)}}' "$SAM_TXT" \
  > "$OUT_DIR/seq4.txt"

# 4) Histograms + diversity metric
python3 /home/neeyuhuynh/Desktop/me/genesolution/q4_solution/hist_col9.py \
  "$OUT_DIR/col9_abs.txt" "$OUT_DIR/col9_hist.csv" "$OUT_DIR/col9_hist.png"

python3 /home/neeyuhuynh/Desktop/me/genesolution/q4_solution/hist_seq4.py \
  "$OUT_DIR/seq4.txt" "$OUT_DIR/seq4_hist.csv" "$OUT_DIR/seq4_hist.png" \
  "$OUT_DIR/seq4_diversity.txt"

echo "Done. Outputs in $OUT_DIR"
