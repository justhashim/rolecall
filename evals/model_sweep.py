"""Which embedding model? Measured, not assumed.

    python -m evals.model_sweep [model ...]

Scores each model alone (vector) and fused with Postgres keyword search (hybrid, RRF), on
the same snapshot and labels as retrieval_eval. Vectors are brute-forced in numpy so models
of any dimension can be compared without rebuilding the table. Document vectors are cached
in .cache/sweep/.
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np

from rolecall import embed, search

from .retrieval_eval import K, _conn, mrr, ndcg, qrels, queries

CACHE = Path(".cache/sweep")
MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
    "snowflake/snowflake-arctic-embed-s",
    "minishlab/potion-multilingual-128M",
    "jinaai/jina-embeddings-v2-base-de",
    "google/embeddinggemma-300m",
    "intfloat/multilingual-e5-large",
]


def doc_vectors(model: str, rows: list[dict]) -> tuple[np.ndarray, float]:
    path = CACHE / (model.replace("/", "__") + ".npy")
    if path.exists():
        return np.load(path), float("nan")
    t = time.perf_counter()
    vecs = embed.embed([embed.job_text(r) for r in rows], model=model)
    secs = time.perf_counter() - t
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs, secs


def main(models: list[str]) -> None:
    conn = _conn()
    rows = conn.execute("select id, source_id, title, tags, location, description from jobs order by id").fetchall()
    ids = np.array([r["id"] for r in rows])
    sid = {r["id"]: r["source_id"] for r in rows}
    labels = qrels()
    keyword = {q["qid"]: search.keyword_ids(conn, q["query"], search.Filters()) for q in queries()}

    print(f"{'model':<62}{'vec nDCG':>9}{'hyb nDCG':>9}{'hyb MRR':>8}{'q ms':>7}{'index s':>9}")
    for model in models:
        vecs, index_secs = doc_vectors(model, rows)
        v_ndcg, h_ndcg, h_mrr, q_ms = [], [], [], []
        for q in queries():
            t = time.perf_counter()
            qv = embed.embed_one(q["query"], model=model)
            q_ms.append((time.perf_counter() - t) * 1000)
            vec_ids = ids[np.argsort(-(vecs @ qv))[: search.CANDIDATES]].tolist()
            hyb_ids = search.rrf(keyword[q["qid"]], vec_ids)
            got = labels.get(q["qid"], {})
            v_ndcg.append(ndcg([sid[i] for i in vec_ids[:K]], got))
            h_ndcg.append(ndcg([sid[i] for i in hyb_ids[:K]], got))
            h_mrr.append(mrr([sid[i] for i in hyb_ids[:K]], got))
        print(f"{model:<62}{statistics.mean(v_ndcg):>9.3f}{statistics.mean(h_ndcg):>9.3f}{statistics.mean(h_mrr):>8.3f}"
              f"{statistics.median(q_ms):>7.0f}{index_secs:>9.0f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
