#!/usr/bin/env bash
# Continuous GPU sampler -> CSV timeline (run on the L4, in its own terminal).
# The benchmark also samples GPU per-config; this gives you a full timeline you
# can plot for the writeup.
#
#   bash bench/gpu_sample.sh results/gpu_timeline.csv
set -euo pipefail
OUT="${1:-results/gpu_timeline.csv}"
mkdir -p "$(dirname "$OUT")"
echo "timestamp,gpu_util,mem_used_mb" > "$OUT"
echo "sampling nvidia-smi every 1s -> $OUT (Ctrl-C to stop)"
while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader,nounits >> "$OUT"
  sleep 1
done
