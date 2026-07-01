#!/usr/bin/env bash
# Serve a SINGLE embedding model with vLLM (manual testing / sanity check).
#
# The benchmark (bench/benchmark.py) manages the vLLM lifecycle itself and
# restarts it per model, so you do NOT need this for the sweep. Use it only to
# poke a single model by hand, e.g. before running --no-serve.
#
#   bash server/run_vllm.sh BAAI/bge-base-en-v1.5
#
# If your vLLM version rejects `--task embed`, try `--task embedding`.
set -euo pipefail
MODEL="${1:-BAAI/bge-base-en-v1.5}"
PORT="${2:-8000}"
TASK="${3:-embed}"

echo "serving $MODEL on :$PORT (task=$TASK)"
exec vllm serve "$MODEL" --task "$TASK" --port "$PORT"
