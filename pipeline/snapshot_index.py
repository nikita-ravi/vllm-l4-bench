"""Snapshot-time index compilation.

The TrueNAS phrasing: "snapshot-time index compilation." When a ZFS snapshot is
taken, you compile every per-file sidecar into one queryable index. Here that is
exactly: gather all *.emb.npy sidecars -> one FAISS index.

  python pipeline/snapshot_index.py --sidecars drop --out results/snapshot.faiss
"""
from __future__ import annotations

import argparse

from index import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecars", default="drop")
    ap.add_argument("--out", default="results/snapshot.faiss")
    a = ap.parse_args()
    print("compiling snapshot index from sidecars ...")
    build(a.sidecars, a.out)


if __name__ == "__main__":
    main()
