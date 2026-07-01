"""Build a FAISS index from sidecar embeddings and run similarity queries.

In the appliance this is where cuVS / CAGRA would sit instead of FAISS — a
GPU-native ANN index with the same interface, keeping search on the GPU next to
the embedding model. FAISS-CPU here is the local stand-in.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import faiss

from embed_client import EmbedClient, as_query, MODELS


def load_sidecars(sidecar_dir):
    paths = sorted(glob.glob(os.path.join(sidecar_dir, "**", "*.emb.npy"), recursive=True))
    vecs, names = [], []
    for p in paths:
        vecs.append(np.load(p))
        names.append(os.path.basename(p)[: -len(".emb.npy")])
    if not vecs:
        raise SystemExit(f"No .emb.npy sidecars found under {sidecar_dir}")
    mat = np.vstack(vecs).astype(np.float32)
    faiss.normalize_L2(mat)
    return mat, names


def build(sidecar_dir, out_path):
    mat, names = load_sidecars(sidecar_dir)
    index = faiss.IndexFlatIP(mat.shape[1])  # inner product on L2-normalized = cosine
    index.add(mat)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    faiss.write_index(index, out_path)
    with open(out_path + ".names.json", "w") as f:
        json.dump(names, f)
    print(f"Indexed {len(names)} vectors (dim={mat.shape[1]}) -> {out_path}")
    return index, names


def search(index, names, qvec, k=5):
    q = qvec.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, min(k, index.ntotal))
    return [(names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1]


def _load(index_path):
    index = faiss.read_index(index_path)
    with open(index_path + ".names.json") as f:
        names = json.load(f)
    return index, names


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build FAISS index from sidecars")
    b.add_argument("--sidecars", default="drop")
    b.add_argument("--out", default="results/index.faiss")

    q = sub.add_parser("query", help="embed a query and return nearest files")
    q.add_argument("--index", default="results/index.faiss")
    q.add_argument("--text", required=True)
    q.add_argument("--model", default="BAAI/bge-base-en-v1.5", choices=list(MODELS))
    q.add_argument("--base-url", default="http://localhost:8000")
    q.add_argument("--stub", action="store_true")
    q.add_argument("-k", type=int, default=5)

    a = ap.parse_args()
    if a.cmd == "build":
        build(a.sidecars, a.out)
    elif a.cmd == "query":
        index, names = _load(a.index)
        client = EmbedClient(a.model, base_url=a.base_url, stub=a.stub)
        qvec = client.embed_sync([as_query(a.text)])[0]
        print(f"query: {a.text!r}\n")
        for name, score in search(index, names, qvec, a.k):
            print(f"  {score:6.3f}  {name}")


if __name__ == "__main__":
    main()
