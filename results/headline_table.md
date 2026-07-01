# Serving-configuration benchmark: BGE embedding models on one GPU

| Model | dim | batch | conc | throughput (emb/s) | p50 ms | p99 ms | GPU util | VRAM (load) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | 384 | 1 | 1 | 391 | 1.8 | 1.8 | n/a | n/a |
| bge-small-en-v1.5 | 384 | 1 | 8 | 1652 | 1.9 | 2.5 | n/a | n/a |
| bge-small-en-v1.5 | 384 | 16 | 1 | 610 | 25.9 | 26.7 | n/a | n/a |
| bge-small-en-v1.5 | 384 | 16 | 8 | 3852 | 27.5 | 29.8 | n/a | n/a |
| bge-base-en-v1.5 | 768 | 1 | 1 | 529 | 1.8 | 1.8 | n/a | n/a |
| bge-base-en-v1.5 | 768 | 1 | 8 | 2241 | 2.0 | 2.3 | n/a | n/a |
| bge-base-en-v1.5 | 768 | 16 | 1 | 613 | 25.8 | 26.5 | n/a | n/a |
| bge-base-en-v1.5 | 768 | 16 | 8 | 3813 | 27.8 | 30.0 | n/a | n/a |
| bge-large-en-v1.5 | 1024 | 1 | 1 | 439 | 1.9 | 2.4 | n/a | n/a |
| bge-large-en-v1.5 | 1024 | 1 | 8 | 1925 | 1.9 | 2.2 | n/a | n/a |
| bge-large-en-v1.5 | 1024 | 16 | 1 | 604 | 26.3 | 26.6 | n/a | n/a |
| bge-large-en-v1.5 | 1024 | 16 | 8 | 3654 | 28.7 | 31.1 | n/a | n/a |

_VRAM (load) = GPU memory used right after the model became healthy, before load. Headroom = 24 GB (L4) minus this — the budget left for storage workloads to coexist on the same appliance._
