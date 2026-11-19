"""Local embeddings. No API key, no per-call cost, same vectors every run."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# fastembed defaults to /tmp, which is often a small tmpfs. Models are ~0.2 to 0.6 GB.
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))

# Multilingual on purpose: most of the feed is German, most queries are English.
# evals/retrieval_eval.py compares it against an English-only model of the same size.
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384
# The model truncates at 128 word pieces, so only the start of the description counts.
DESCRIPTION_CHARS = 600


def job_text(job: dict) -> str:
    tags = ", ".join(job.get("tags") or [])
    return f"{job['title']}. {tags}. {job.get('location', '')}. {job.get('description', '')[:DESCRIPTION_CHARS]}"


@lru_cache(maxsize=4)
def _model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=CACHE_DIR)


def _normalise(vecs) -> np.ndarray:
    vecs = np.array(list(vecs), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed(texts: list[str], model: str = MODEL) -> np.ndarray:
    """Documents. passage_embed adds whatever prefix the model was trained with (e5, gemma...)."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return _normalise(_model(model).passage_embed(texts, batch_size=32))


def embed_one(text: str, model: str = MODEL) -> np.ndarray:
    """A search query. Asymmetric models embed queries differently from documents."""
    return _normalise(_model(model).query_embed(text))[0]
