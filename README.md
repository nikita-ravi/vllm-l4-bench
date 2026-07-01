# On-array vectorization pipeline + serving-configuration benchmark

A small, honest demonstration of the GPU-inference decisions a storage appliance
has to make: **turn files into embeddings as they land, index them for search,
and measure what each serving configuration costs in throughput, latency, and
VRAM** — so you know how much GPU headroom is left for storage workloads.

Built to mirror three responsibilities directly:

| This repo | The responsibility it mirrors |
|---|---|
| `pipeline/watcher.py` | per-file **sidecar generation driven by filesystem events** |
| `pipeline/snapshot_index.py` | **snapshot-time index compilation** |
| `bench/benchmark.py` | **model selection + quantization/serving tradeoffs** on L4/A2-class GPUs |
| FAISS today, **cuVS/CAGRA** note in `pipeline/index.py` | GPU-accelerated vector search |

## Architecture

```
file lands ─▶ watcher ─▶ POST /v1/embeddings ─▶ vLLM (L4 / stub) ─▶ <file>.emb.npy sidecar
                                                                          │
                                          snapshot_index ─▶ FAISS index ◀─┘ ─▶ top-k query
benchmark ─▶ per model: start vLLM ▸ capture VRAM ▸ sweep batch×concurrency ▸ stop ▸ table
```

The whole thing is built around **one embedding client** (`pipeline/embed_client.py`)
that talks to a stub, a local vLLM, or an L4 vLLM — only the base URL changes.

## Two legs

**Mac (dev / validate, no GPU).** vLLM's fast path is CUDA-only, so the Mac leg
proves the pipeline and the benchmark orchestration are correct using `--stub`
(deterministic pseudo-embeddings, no server).

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-mac.txt

python scripts/make_samples.py                       # sample corpus in drop/
python pipeline/watcher.py --dir drop --stub --once  # files -> sidecars
python pipeline/snapshot_index.py --sidecars drop --out results/index.faiss
python pipeline/index.py query --index results/index.faiss --stub \
    --text "how do copy-on-write snapshots work"     # similarity search
python bench/benchmark.py --stub --quick             # validate the sweep logic
```

**L4 (real numbers).** Rent an NVIDIA L4 (~$0.40–0.80/hr). The benchmark manages
vLLM itself — starts each model, captures load VRAM, sweeps, stops, writes the table.

```bash
pip install -r requirements-l4.txt
# terminal 1: continuous GPU timeline
bash bench/gpu_sample.sh results/gpu_timeline.csv
# terminal 2: full multi-model sweep -> results/headline_table.md
python bench/benchmark.py --n-texts 2000
```

Same client code, real GPU. Tear the box down when done (~15–30 min of runtime).

## What comes out

`results/headline_table.md` — one row per **model × batch × concurrency**
(3 models × 4 batches × 3 concurrency = 36 configs), with throughput, p50/p99
latency, GPU utilization, and load-time VRAM.

**The headline sentence** (fill from your run): *"On an L4, batching to 64 lifted
embedding throughput ~N× over batch-1; bge-large gave 2.6× the vector dimension of
bge-small for ~M× the throughput cost; the largest model used only ~V GB of 24 GB,
leaving ~H GB of headroom to run embedding inference alongside storage workloads on
one appliance."*

## Files
```
pipeline/embed_client.py    one client: stub | local | L4  (BGE query prefix baked in)
pipeline/watcher.py         filesystem events -> .emb.npy sidecars
pipeline/index.py           FAISS build + query  (cuVS/CAGRA swap-in note)
pipeline/snapshot_index.py  compile all sidecars into one index
bench/benchmark.py          multi-model serving-config sweep -> comparison table
bench/gpu_sample.sh         nvidia-smi timeline
server/run_vllm.sh          serve one model by hand (optional)
scripts/make_samples.py     sample corpus
```
