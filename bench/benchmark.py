"""Multi-model serving-configuration benchmark for vLLM embedding models.

Makes MODEL a first-class sweep dimension. For each model it (optionally) starts
vLLM, captures load-time VRAM, sweeps batch x concurrency measuring throughput /
latency / GPU utilization, then stops vLLM — and writes one unified
model x config comparison table.

  # L4 (real run): benchmark manages the vLLM lifecycle itself
  python bench/benchmark.py --n-texts 2000

  # Mac dry-run: validate ALL the orchestration logic offline, no GPU, no vLLM
  python bench/benchmark.py --stub --quick

  # Point at an already-running server (you started vLLM yourself)
  python bench/benchmark.py --no-serve --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from embed_client import EmbedClient, MODELS  # noqa: E402


# ---------- GPU sampling (no-op off-GPU, e.g. on the Mac) ----------
def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        u, m = out.strip().splitlines()[0].split(",")
        return float(u), float(m)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


async def sample_gpu(stop_evt):
    utils, mems = [], []
    while not stop_evt.is_set():
        s = nvidia_smi()
        if s:
            utils.append(s[0])
            mems.append(s[1])
        await asyncio.sleep(0.25)
    if utils:
        return statistics.mean(utils), max(mems)
    return None, None


# ---------- vLLM lifecycle ----------
def wait_health(port, timeout=600):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(2)
    return False


def start_vllm(model, port, task):
    print(f"  starting vLLM: {model} (task={task}, port={port}) ...", flush=True)
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--task", task, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    if not wait_health(port):
        proc.terminate()
        raise SystemExit(f"vLLM for {model} never became healthy")
    print("  vLLM healthy.", flush=True)
    return proc


def stop_vllm(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------- corpus ----------
def make_corpus(n, corpus_dir=None):
    if corpus_dir and os.path.isdir(corpus_dir):
        texts = []
        for root, _, files in os.walk(corpus_dir):
            for fn in files:
                if fn.endswith((".txt", ".md")):
                    with open(os.path.join(root, fn), errors="ignore") as f:
                        texts.append(f.read().strip())
        if texts:
            return (texts * (n // len(texts) + 1))[:n]
    # synthetic: varied-length lines so batching has something realistic to chew
    topics = ["storage array", "zfs snapshot", "gpu inference", "vector search",
              "anomaly detection", "embedding model", "raid resilver", "telemetry log"]
    out = []
    for i in range(n):
        reps = 3 + (i % 30)  # length varies 3..32 tokens-ish
        out.append(" ".join(topics[(i + j) % len(topics)] for j in range(reps)))
    return out


# ---------- one (model, batch, concurrency) measurement ----------
async def run_config(client, corpus, batch, concurrency, n_requests):
    payloads = [corpus[(i * batch) % len(corpus): (i * batch) % len(corpus) + batch]
                for i in range(n_requests)]
    payloads = [p if len(p) == batch else (corpus[:batch]) for p in payloads]

    latencies = []
    sem = asyncio.Semaphore(concurrency)

    async def one(payload):
        async with sem:
            t0 = time.perf_counter()
            await client.embed(payload)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    stop_evt = asyncio.Event()
    gpu_task = asyncio.create_task(sample_gpu(stop_evt))

    start = time.perf_counter()
    await asyncio.gather(*(one(p) for p in payloads))
    elapsed = time.perf_counter() - start

    stop_evt.set()
    gpu_util, gpu_mem = await gpu_task

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
    p99 = lat_sorted[int(0.99 * (len(lat_sorted) - 1))]
    total_emb = n_requests * batch
    return {
        "batch": batch, "concurrency": concurrency,
        "throughput": total_emb / elapsed,
        "p50_ms": p50, "p99_ms": p99,
        "gpu_util": gpu_util, "gpu_mem_mb": gpu_mem,
    }


# ---------- driver ----------
async def bench_model(model, base_url, stub, corpus, batches, concs, n_requests):
    client = EmbedClient(model, base_url=base_url, stub=stub)
    rows = []
    for batch in batches:
        for conc in concs:
            r = await run_config(client, corpus, batch, conc, n_requests)
            r["model"] = model
            r["dim"] = MODELS.get(model, 0)
            rows.append(r)
            gu = f"{r['gpu_util']:.0f}%" if r["gpu_util"] is not None else "n/a"
            print(f"    b={batch:>3} c={conc:>3}  {r['throughput']:8.0f} emb/s  "
                  f"p50={r['p50_ms']:6.1f}ms  p99={r['p99_ms']:6.1f}ms  gpu={gu}")
    await client.aclose()
    return rows


def write_outputs(rows, out_md, load_vram):
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    raw = os.path.splitext(out_md)[0] + "_raw.csv"
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dim", "batch", "concurrency",
                                          "throughput", "p50_ms", "p99_ms",
                                          "gpu_util", "gpu_mem_mb"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    with open(out_md, "w") as f:
        f.write("# Serving-configuration benchmark: BGE embedding models on one GPU\n\n")
        f.write("| Model | dim | batch | conc | throughput (emb/s) | p50 ms | p99 ms | GPU util | VRAM (load) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            short = r["model"].split("/")[-1]
            gu = f"{r['gpu_util']:.0f}%" if r["gpu_util"] is not None else "n/a"
            vram = load_vram.get(r["model"])
            vram_s = f"{vram/1024:.1f} GB" if vram else "n/a"
            f.write(f"| {short} | {r['dim']} | {r['batch']} | {r['concurrency']} | "
                    f"{r['throughput']:.0f} | {r['p50_ms']:.1f} | {r['p99_ms']:.1f} | {gu} | {vram_s} |\n")
        f.write("\n_VRAM (load) = GPU memory used right after the model became healthy, "
                "before load. Headroom = 24 GB (L4) minus this — the budget left for "
                "storage workloads to coexist on the same appliance._\n")
    print(f"\nwrote {out_md}\nwrote {raw}")


async def main_async(a):
    models = a.models or list(MODELS)
    batches = [1, 16] if a.quick else [int(x) for x in a.batches.split(",")]
    concs = [1, 8] if a.quick else [int(x) for x in a.concurrency.split(",")]
    n_requests = 30 if a.quick else a.requests_per_config
    n_texts = 200 if a.quick else a.n_texts
    corpus = make_corpus(n_texts, a.corpus_dir)

    manage = (not a.stub) and (not a.no_serve)
    all_rows, load_vram = [], {}
    for model in models:
        print(f"\n=== {model} ===")
        proc = None
        if manage:
            proc = start_vllm(model, a.port, a.task)
            s = nvidia_smi()
            if s:
                load_vram[model] = s[1]
                print(f"  load VRAM: {s[1]/1024:.1f} GB")
        try:
            rows = await bench_model(model, a.base_url, a.stub, corpus,
                                     batches, concs, n_requests)
            all_rows.extend(rows)
        finally:
            if proc is not None:
                stop_vllm(proc)

    write_outputs(all_rows, a.out, load_vram)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=None,
                    help="default: all three BGE models")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--requests-per-config", type=int, default=200)
    ap.add_argument("--n-texts", type=int, default=2000)
    ap.add_argument("--corpus-dir", default=None, help="use real files instead of synthetic")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--task", default="embed", help="vLLM task flag (embed|embedding)")
    ap.add_argument("--stub", action="store_true", help="Mac: no GPU/vLLM, deterministic pseudo-embeddings")
    ap.add_argument("--no-serve", action="store_true", help="assume vLLM already running; don't manage it")
    ap.add_argument("--quick", action="store_true", help="tiny sweep to validate logic fast")
    ap.add_argument("--out", default="results/headline_table.md")
    a = ap.parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
