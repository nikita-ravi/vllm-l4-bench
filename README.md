# vllm-l4-bench — a hands-on look at serving embedding models with vLLM

A hands-on project measuring how an LLM serving engine behaves on a real GPU. I
turned files into embeddings, searched them, and benchmarked the embedding server
under different settings to see what each choice costs in throughput, latency, and
GPU memory.

Everything here ran on a rented **NVIDIA L4** GPU.

---

## What is vLLM (in one paragraph)

When you "serve" an AI model, requests arrive one at a time, but a GPU is happiest
doing lots of work at once. **vLLM** is a server that sits in front of a model and
is very good at packing many requests together so the GPU stays busy — that's what
makes it fast. You start it with one command (`vllm serve <model>`), and it gives
you a web API you can send text to and get answers back. Here I used it to serve
**embedding** models — models that turn a piece of text into a list of numbers that
captures its meaning, so you can search by meaning instead of keywords.

---

## This project is about *inference serving* — here's exactly what that means

"Inference serving" = running a model behind a live API so requests come in and
answers go out, and doing that **efficiently under load** — as opposed to running
the model once in a script. That "under load" part is the whole game, and it's what
this project measures. Concretely, I:

1. **Ran a real inference server.** `vllm serve <model>` loaded the model onto the
   L4 and exposed an HTTP API (`POST /v1/embeddings`). It really goes live —
   *"Application startup complete… Route: /v1/embeddings"*.
2. **Sent real requests to it.** The benchmark opened HTTP connections and fired
   batches of text at that endpoint; the GPU computed the embeddings and sent them
   back. **Every number in the results table is a real request → GPU → response
   round-trip** — nothing simulated.
3. **Measured serving behavior** — throughput, p50/p99 latency, GPU utilization,
   VRAM. These only mean something *while serving*; they're not model-quality
   scores, they're "how does this hold up under traffic" numbers.
4. **Varied the serving configuration** — batch size and concurrency. These are
   *serving* knobs: they don't change the model, they change how requests get
   scheduled onto the GPU. Under the hood, vLLM's **continuous batching** packs
   concurrent requests together to keep the GPU busy — and the 22× batching result
   below is that mechanism being caught in the act.

So the short version: I stood up a production-style inference server and
load-tested it across 36 configurations.

---

## Why I did this

A GPU is expensive and shared. If you're going to run inference on one, you have to
make choices:

- **Which model?** A bigger model is "smarter" but slower and uses more memory.
- **How do you batch requests?** Send them one at a time, or bundle them?
- **How much load can it take** before it slows to a crawl?

I wanted to stop guessing and get real numbers for these tradeoffs on real hardware.

---

## What I tried to find out

I served three sizes of the same embedding model (BGE **small**, **base**, and
**large**) and, for each one, swept two knobs:

- **batch size** — how many texts I send in one request (1, 8, 32, 64)
- **concurrency** — how many requests are in flight at once (1, 8, 32)

For every combination I measured: **throughput** (texts embedded per second),
**latency** (how long a request takes — p50 = typical, p99 = worst case),
**GPU utilization**, and **GPU memory used**. That's 3 models × 4 × 3 = **36
configurations**, all in one automated run.

The questions behind it:
1. How much does batching actually help?
2. What does choosing a bigger model really cost?
3. How much of the GPU is left over — could storage work run alongside it?
4. At what point does piling on load make latency blow up?

---

## What I found

Full table: **[results/headline_table.md](results/headline_table.md)**. The short
version:

**1. Batching is the single biggest win.**
Sending texts one at a time barely used the GPU. Bundling them up took the small
model from **71 → 1,591 texts/sec** — about **22× faster**, same hardware, same
model. The whole reason vLLM exists, seen in one number.

**2. A bigger model costs roughly what you'd expect — now quantified.**
Peak speed dropped **1,591 → 1,175 → 835 texts/sec** going small → base → large.
So the large model is ~1.9× slower but gives a 2.67× richer vector (1024 numbers
vs 384). That's the "is the extra quality worth it?" decision, in actual numbers.

**3. Embeddings barely tax an L4 — lots of headroom.**
Even the large model used only **1.2 GB out of 24 GB**, and the small/base models
kept GPU utilization under ~27%. Meaning: you could run this embedding work on the
same box as other workloads (like a storage system) and still have most of the GPU
free.

**4. There's a clear "too much load" cliff.**
Latency stays low, then suddenly explodes. Push the large model to batch 64 with 32
concurrent requests and the worst-case latency hits **10.5 seconds** — the point
where requests start piling up in a queue. Good to know where that line is before
you cross it in production.

---

## Try it yourself

**On a GPU box (I used a RunPod L4):**
```bash
pip install -r requirements-l4.txt
python bench/benchmark.py --n-texts 2000      # runs all 3 models, writes the table
```

**On a Mac (no GPU) — run the pipeline without a GPU:**
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-mac.txt
python scripts/make_samples.py                        # make some sample files
python pipeline/watcher.py --dir drop --stub --once   # files -> embeddings
python pipeline/snapshot_index.py --sidecars drop --out results/index.faiss
python pipeline/index.py query --index results/index.faiss --stub \
    --text "how do copy-on-write snapshots work"      # search by meaning
python bench/benchmark.py --stub --quick              # dry-run the benchmark logic
```
(`--stub` fakes the embeddings so it runs without a GPU — real numbers need the L4.)

---

## Getting it running on the L4

Standing up vLLM on the L4 took solving a chain of version constraints. Notes are
in `requirements-l4.txt`; the short list:

1. The newest `pip install vllm` grabbed a **CUDA-13 build** the L4's driver was
   too old for → pinned **vllm==0.11.0** (CUDA-12 build).
2. That vLLM needed the older **transformers 4.x** API → `transformers<5`.
3. The box had fast-downloads switched on but was **missing `hf_transfer`**, so
   model downloads failed → installed it.
4. vLLM **renamed the embedding flag** between versions (`--task embed` on 0.11,
   `--runner pooling` on newer ones).

---

## What's in here

```
pipeline/   turn files into embeddings (watcher), search them (FAISS)
bench/      the multi-model benchmark that produced the table
server/     run one model by hand (optional)
scripts/    make sample files
results/    the actual numbers from the L4 run
```
