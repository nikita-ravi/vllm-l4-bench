"""Filesystem-event-driven embedding: a file lands -> an .emb.npy sidecar appears.

This mirrors the TrueNAS responsibility verbatim: "per-file sidecar generation
driven by filesystem events." Point it at a directory, drop text files in, and a
sidecar embedding is written next to each one.

  python pipeline/watcher.py --dir drop --stub --once      # process existing, exit
  python pipeline/watcher.py --dir drop --base-url http://localhost:8000   # watch
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from embed_client import EmbedClient, MODELS

DEFAULT_EXTS = (".txt", ".md")


def sidecar_path(path):
    return path + ".emb.npy"


def needs_embed(path):
    side = sidecar_path(path)
    return not (os.path.exists(side) and os.path.getmtime(side) >= os.path.getmtime(path))


def process_file(client, path, exts):
    if not path.endswith(tuple(exts)):
        return
    if not needs_embed(path):
        return
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read().strip()
    except (FileNotFoundError, IsADirectoryError):
        return
    if not text:
        return
    vec = client.embed_sync([text])[0]
    np.save(sidecar_path(path), vec)
    print(f"  embedded {os.path.basename(path)} -> {os.path.basename(sidecar_path(path))} (dim={vec.shape[0]})")


class SidecarHandler(FileSystemEventHandler):
    def __init__(self, client, exts):
        self.client = client
        self.exts = exts

    def on_created(self, event):
        if not event.is_directory:
            process_file(self.client, event.src_path, self.exts)

    def on_modified(self, event):
        if not event.is_directory:
            process_file(self.client, event.src_path, self.exts)


def backfill(client, directory, exts):
    files = [p for p in glob.glob(os.path.join(directory, "**", "*"), recursive=True)
             if os.path.isfile(p) and p.endswith(tuple(exts))]
    print(f"backfill: {len(files)} candidate file(s) in {directory}")
    for p in sorted(files):
        process_file(client, p, exts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="drop")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5", choices=list(MODELS))
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--stub", action="store_true", help="no server; deterministic pseudo-embeddings")
    ap.add_argument("--exts", nargs="+", default=list(DEFAULT_EXTS))
    ap.add_argument("--once", action="store_true", help="backfill existing files then exit")
    a = ap.parse_args()

    os.makedirs(a.dir, exist_ok=True)
    client = EmbedClient(a.model, base_url=a.base_url, stub=a.stub)
    print(f"model={a.model} stub={a.stub} dir={a.dir}")

    backfill(client, a.dir, a.exts)
    if a.once:
        return

    handler = SidecarHandler(client, a.exts)
    obs = Observer()
    obs.schedule(handler, a.dir, recursive=True)
    obs.start()
    print(f"watching {a.dir} for {a.exts} files ... (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


if __name__ == "__main__":
    main()
