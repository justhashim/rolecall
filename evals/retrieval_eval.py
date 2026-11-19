"""Retrieval eval: which search mode actually finds the right jobs?

    python -m evals.retrieval_eval pool   # dump the pooled top 10s for labelling
    python -m evals.retrieval_eval run    # score every system against qrels.jsonl

Labels are graded: 2 = what the query asked for, 1 = related (adjacent role or seniority),
0 = wrong. Unlabelled results count as 0. Pooling is the standard TREC shortcut: only the
union of every system's top 10 gets labelled, so recall is relative to that pool.
"""

import json
import math
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from rolecall import embed, search

from . import snapshot

DATA = Path(__file__).parent / "data"
QRELS = DATA / "qrels.jsonl"
RESULTS = Path(__file__).parent / "results"
K = 10
ENGLISH_MODEL = "BAAI/bge-small-en-v1.5"  # same size and dim as the multilingual one


def queries() -> list[dict]:
    return [json.loads(l) for l in (DATA / "queries.jsonl").read_text().splitlines() if l.strip()]


def qrels() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for line in QRELS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["qid"], {})[r["source_id"]] = r["grade"]
    return out


@lru_cache(maxsize=1)
def _english_index(conn_id: int):
    """English-only ablation, brute force in numpy. Not a production path."""
    conn = _conn()
    rows = conn.execute("select id, title, tags, location, description from jobs order by id").fetchall()
    vecs = embed.embed([embed.job_text(r) for r in rows], model=ENGLISH_MODEL)
    return np.array([r["id"] for r in rows]), vecs


@lru_cache(maxsize=1)
def _conn():
    return snapshot.load()


def run_system(name: str, query: str) -> list[int]:
    conn = _conn()
    if name == "vector_english":
        ids, vecs = _english_index(0)
        scores = vecs @ embed.embed_one(query, model=ENGLISH_MODEL)
        return ids[np.argsort(-scores)[:K]].tolist()
    mode, rerank = {
        "keyword": ("keyword", False),
        "vector": ("vector", False),
        "hybrid": ("hybrid", False),
        "hybrid_rerank": ("hybrid", True),
    }[name]
    return search.search_ids(conn, query, mode=mode, limit=K, use_rerank=rerank)


SYSTEMS = ["keyword", "vector_english", "vector", "hybrid", "hybrid_rerank"]


def source_ids(ids: list[int]) -> list[str]:
    rows = {r["id"]: r["source_id"] for r in _conn().execute("select id, source_id from jobs where id = any(%s)", (ids,))}
    return [rows[i] for i in ids]


def ndcg(ranked: list[str], labels: dict[str, int], k: int = K) -> float:
    dcg = sum((2 ** labels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[str], labels: dict[str, int], k: int = K) -> float:
    relevant = {d for d, g in labels.items() if g >= 1}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], labels: dict[str, int]) -> float:
    """Reciprocal rank of the first result that's exactly what was asked for (grade 2)."""
    return next((1 / (i + 1) for i, d in enumerate(ranked) if labels.get(d, 0) == 2), 0.0)


def pool() -> None:
    conn = _conn()
    for q in queries():
        ids: dict[int, None] = {}
        for system in SYSTEMS:
            ids.update(dict.fromkeys(run_system(system, q["query"])))
        rows = {r["id"]: r for r in conn.execute("select id, source_id, title, company, tags from jobs where id = any(%s)", (list(ids),))}
        for i in ids:
            r = rows[i]
            print(json.dumps({"qid": q["qid"], "query": q["query"], "source_id": r["source_id"], "title": r["title"],
                              "company": r["company"], "tags": r["tags"][:4]}, ensure_ascii=False))


def run() -> None:
    labels = qrels()
    rows, per_query = [], {}
    for system in SYSTEMS:
        scores = {"ndcg@10": [], "recall@10": [], "mrr": [], "ms": []}
        for q in queries():
            run_system(system, q["query"])  # warm caches so latency is search, not model load
            t = time.perf_counter()
            ranked = source_ids(run_system(system, q["query"]))
            scores["ms"].append((time.perf_counter() - t) * 1000)
            got = labels.get(q["qid"], {})
            scores["ndcg@10"].append(ndcg(ranked, got))
            scores["recall@10"].append(recall(ranked, got))
            scores["mrr"].append(mrr(ranked, got))
            per_query.setdefault(q["qid"], {})[system] = round(scores["ndcg@10"][-1], 3)
        rows.append({
            "system": system,
            **{m: round(statistics.mean(v), 3) for m, v in scores.items() if m != "ms"},
            "p50_ms": round(statistics.median(scores["ms"]), 1),
        })

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps({"summary": rows, "ndcg_by_query": per_query}, indent=2))
    print(f"{'system':<16}{'nDCG@10':>9}{'Recall@10':>11}{'MRR':>7}{'p50 ms':>9}")
    for r in rows:
        print(f"{r['system']:<16}{r['ndcg@10']:>9.3f}{r['recall@10']:>11.3f}{r['mrr']:>7.3f}{r['p50_ms']:>9.1f}")


if __name__ == "__main__":
    {"pool": pool, "run": run}[sys.argv[1]]()
