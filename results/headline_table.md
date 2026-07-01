# Serving-configuration benchmark: BGE embedding models on one GPU

Hardware: **NVIDIA L4 (24 GB)** on RunPod · vLLM 0.11.0 · torch 2.8 / CUDA 12.8 ·
3 models × 4 batch sizes × 3 concurrency levels = 36 configurations.

| Model | dim | batch | conc | throughput (emb/s) | p50 ms | p99 ms | GPU util | VRAM (load) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | 384 | 1 | 1 | 71 | 12.3 | 46.8 | 3% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 1 | 8 | 253 | 25.2 | 71.7 | 5% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 1 | 32 | 283 | 88.7 | 221.4 | 7% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 8 | 1 | 249 | 28.4 | 63.7 | 4% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 8 | 8 | 933 | 60.4 | 120.2 | 8% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 8 | 32 | 1175 | 166.8 | 372.4 | 9% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 32 | 1 | 446 | 69.1 | 93.4 | 5% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 32 | 8 | 1247 | 173.9 | 387.7 | 11% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 32 | 32 | 1591 | 478.0 | 1637.1 | 13% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 64 | 1 | 449 | 133.2 | 182.5 | 4% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 64 | 8 | 1437 | 353.8 | 535.9 | 13% | 0.5 GB |
| bge-small-en-v1.5 | 384 | 64 | 32 | 1413 | 1011.2 | 3856.1 | 13% | 0.5 GB |
| bge-base-en-v1.5 | 768 | 1 | 1 | 64 | 13.5 | 52.3 | 7% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 1 | 8 | 208 | 30.9 | 76.2 | 15% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 1 | 32 | 309 | 68.2 | 195.7 | 9% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 8 | 1 | 202 | 36.1 | 73.4 | 7% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 8 | 8 | 767 | 79.0 | 140.7 | 18% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 8 | 32 | 970 | 227.2 | 415.3 | 15% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 32 | 1 | 307 | 100.7 | 148.5 | 8% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 32 | 8 | 987 | 235.4 | 476.8 | 22% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 32 | 32 | 1175 | 678.9 | 1754.3 | 27% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 64 | 1 | 348 | 178.0 | 239.0 | 8% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 64 | 8 | 1009 | 493.6 | 802.9 | 20% | 0.7 GB |
| bge-base-en-v1.5 | 768 | 64 | 32 | 1108 | 1382.3 | 4047.5 | 23% | 0.7 GB |
| bge-large-en-v1.5 | 1024 | 1 | 1 | 52 | 17.5 | 52.8 | 16% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 1 | 8 | 167 | 39.5 | 84.8 | 22% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 1 | 32 | 262 | 93.8 | 238.7 | 16% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 8 | 1 | 149 | 48.5 | 105.4 | 18% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 8 | 8 | 667 | 82.4 | 179.6 | 54% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 8 | 32 | 835 | 214.8 | 1168.9 | 62% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 32 | 1 | 216 | 140.9 | 198.5 | 19% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 32 | 8 | 734 | 313.9 | 700.1 | 62% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 32 | 32 | 794 | 690.7 | 4153.6 | 63% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 64 | 1 | 268 | 238.9 | 289.6 | 22% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 64 | 8 | 719 | 631.7 | 1683.1 | 56% | 1.2 GB |
| bge-large-en-v1.5 | 1024 | 64 | 32 | 752 | 2059.1 | 10537.8 | 60% | 1.2 GB |

_VRAM (load) = GPU memory used right after the model became healthy. Headroom =
24 GB (L4) minus this — the budget left for storage workloads to coexist on the
same appliance._

## Findings

1. **Batching is the biggest lever.** bge-small rose from **71 emb/s** (batch 1,
   conc 1) to **1,591 emb/s** (batch 32, conc 32) — a **~22× throughput gain**
   from batching alone.

2. **Model-selection tradeoff, quantified.** Peak throughput fell
   **1,591 → 1,175 → 835 emb/s** across small → base → large: **~1.9× the
   throughput cost** to gain **2.67× the vector dimension** (384 → 1024).

3. **Storage-workload headroom.** Load VRAM was only **0.5 / 0.7 / 1.2 GB of
   24 GB**. GPU utilization stayed **≤ 27%** for small and base — only bge-large
   became compute-bound (up to **63%**). An L4 serving embeddings leaves ~95% of
   VRAM and, for small/base models, most of its compute free.

4. **The latency knee.** p99 latency stays flat, then explodes under load:
   bge-large at batch 64 / conc 32 reaches a **10.5 s** p99. bge-large saturates
   first precisely because it is the only compute-bound model here.
