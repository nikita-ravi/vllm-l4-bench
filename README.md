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

## Results (real run, NVIDIA L4)

Full 36-config table + findings: **[results/headline_table.md](results/headline_table.md)**.
Headline numbers from the L4 run:

- **Batching is the biggest lever** — bge-small: **71 → 1,591 emb/s** (~22×) from
  batch 1 to batch 32 at concurrency 32.
- **Model-selection tradeoff** — peak throughput **1,591 → 1,175 → 835 emb/s**
  (small → base → large): ~1.9× throughput cost for 2.67× the vector dimension.
- **Storage-workload headroom** — load VRAM only **0.5 / 0.7 / 1.2 GB of 24 GB**;
  GPU util ≤ 27% for small/base, up to 63% only for large.
- **Latency knee** — p99 explodes under load (bge-large hits **10.5 s** at
  batch 64 / conc 32) — the point past which requests queue.

## Lessons from the real run (RunPod L4)

Reproducing this on a rented L4 surfaced a chain of version constraints worth
knowing — captured in `requirements-l4.txt`:

1. The newest `pip install vllm` pulls a **CUDA-13 torch** that the L4 host driver
   (565 / CUDA 12.7) is too old for → pin **vllm==0.11.0** (torch 2.8 / CUDA 12.8).
2. vLLM 0.11 needs the **transformers 4.x** tokenizer API → `transformers<5`.
3. The image enables `HF_HUB_ENABLE_HF_TRANSFER=1`, so **`hf_transfer`** must be
   installed or uncached model downloads fail mid-run.
4. vLLM renamed the embedding flag across versions: **0.11 uses `--task embed`**
   (deprecated warning is harmless); 0.24+ uses `--runner pooling`.

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
