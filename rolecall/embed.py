"""Local embeddings. No API key, no per-call cost, same vectors every run."""

from functools import lru_cache

import numpy as np

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

    return TextEmbedding(name)


def embed(texts: list[str], model: str = MODEL) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    vecs = np.array(list(_model(model).embed(texts, batch_size=64)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_one(text: str, model: str = MODEL) -> np.ndarray:
    return embed([text], model)[0]
