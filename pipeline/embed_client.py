"""OpenAI-compatible embedding client for vLLM's /v1/embeddings endpoint.

The whole point of this file: ONE client, used everywhere. The same code path
talks to a stub (Mac dev, no GPU), a local vLLM, or an L4 vLLM. Only the base
URL (or --stub) changes between the Mac dev leg and the real L4 run.

BGE retrieval convention: prepend BGE_QUERY_PREFIX to QUERIES only (never to the
stored documents). Skipping this quietly degrades search quality.
"""
from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import httpx

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# model -> embedding dimension. Small/base/large = the model-selection tradeoff.
MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


def as_query(text: str) -> str:
    """Wrap a search query with the BGE retrieval prefix."""
    return BGE_QUERY_PREFIX + text


class EmbedClient:
    """Async + sync embedding client. Async path is used by the benchmark
    (needs real concurrency); sync path is used by the watcher (one file at a
    time). In stub mode, returns deterministic pseudo-embeddings with no server
    so the Mac dry-run exercises all the orchestration logic offline."""

    def __init__(self, model, base_url="http://localhost:8000", stub=False, timeout=120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.stub = stub
        self.timeout = timeout
        self.dim = MODELS.get(model, 768)
        self._aclient = None if stub else httpx.AsyncClient(timeout=timeout)

    # --- async (benchmark) ---
    async def embed(self, texts):
        texts = list(texts)
        if self.stub:
            # simulate per-item compute so batching/concurrency show real effects
            await asyncio.sleep(0.0015 * len(texts))
            return self._stub_embed(texts)
        resp = await self._aclient.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return self._parse(resp.json())

    # --- sync (watcher) ---
    def embed_sync(self, texts):
        texts = list(texts)
        if self.stub:
            return self._stub_embed(texts)
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model, "input": texts},
            )
            r.raise_for_status()
            return self._parse(r.json())

    @staticmethod
    def _parse(payload):
        data = sorted(payload["data"], key=lambda d: d["index"])
        return np.array([d["embedding"] for d in data], dtype=np.float32)

    def _stub_embed(self, texts):
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out

    async def aclose(self):
        if self._aclient is not None:
            await self._aclient.aclose()
