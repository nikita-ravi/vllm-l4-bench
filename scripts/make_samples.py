"""Generate a small, topically-distinct sample corpus in drop/ so the watcher +
FAISS similarity search have something meaningful to retrieve against."""
import os

DOCS = {
    "zfs_snapshot": "ZFS snapshots are read-only point-in-time copies of a dataset. "
                    "Because ZFS is copy-on-write, a snapshot is nearly free to create and "
                    "only consumes space as the live data diverges from it.",
    "zfs_resilver": "Resilvering rebuilds redundancy after a disk is replaced in a ZFS pool. "
                    "Only the blocks that actually exist are copied, so resilver time scales "
                    "with used capacity rather than raw disk size.",
    "gpu_inference": "GPU inference serves a model's forward pass on the GPU. Batching many "
                     "requests together keeps the GPU's compute units busy and raises "
                     "throughput at the cost of some added per-request latency.",
    "paged_attention": "PagedAttention stores the KV cache in fixed-size blocks like virtual "
                       "memory pages, so vLLM can pack many concurrent sequences into GPU "
                       "memory without reserving worst-case contiguous space per request.",
    "embeddings": "An embedding model maps text to a fixed-length vector so that similar "
                  "meanings land near each other. These vectors power semantic search and "
                  "retrieval-augmented generation.",
    "vector_search": "Approximate nearest-neighbor search finds the closest vectors to a query "
                     "without scanning every point. Graph indexes like CAGRA run this search "
                     "directly on the GPU for very low latency.",
    "anomaly_detection": "Time-series anomaly detection flags telemetry that departs from its "
                         "normal pattern. On a storage fleet this surfaces failing drives and "
                         "unusual load before they cause an outage.",
    "quantization": "Quantization stores model weights in fewer bits, such as 4-bit AWQ, to cut "
                    "GPU memory and speed up inference, trading a small amount of accuracy for a "
                    "large reduction in footprint.",
}


def main():
    os.makedirs("drop", exist_ok=True)
    for name, text in DOCS.items():
        path = os.path.join("drop", f"{name}.txt")
        with open(path, "w") as f:
            f.write(text + "\n")
    print(f"wrote {len(DOCS)} sample files to drop/")


if __name__ == "__main__":
    main()
