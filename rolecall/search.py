"""Hybrid retrieval over open jobs.

keyword  Postgres full-text, terms OR-ed so one missing word doesn't zero the result.
vector   pgvector cosine distance on an HNSW index.
hybrid   both lists fused with reciprocal rank fusion (RRF). Rank-based, so the two
         scores never have to be put on the same scale.
rerank   optional cross-encoder pass over the fused top N.

With a user id, saved jobs pull the query vector towards them and skipped jobs push it
away (Rocchio feedback), and skipped jobs are filtered out.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np
import psycopg

from . import embed

Mode = Literal["keyword", "vector", "hybrid"]

RRF_K = 60
CANDIDATES = 50
SAVED_WEIGHT, SKIPPED_WEIGHT = 0.35, 0.2
RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"

_WORD = re.compile(r"[\w+#.]+", re.UNICODE)


@dataclass
class Filters:
    remote: bool | None = None
    location: str | None = None
    exclude_ids: list[int] = field(default_factory=list)

    def sql(self) -> tuple[str, dict]:
        clauses, params = ["closed_at is null"], {}
        if self.remote is not None:
            clauses.append("remote = %(remote)s")
            params["remote"] = self.remote
        if self.location:
            clauses.append("location ilike %(location)s")
            params["location"] = f"%{self.location}%"
        if self.exclude_ids:
            clauses.append("not (id = any(%(exclude_ids)s))")
            params["exclude_ids"] = self.exclude_ids
        return " and ".join(clauses), params


def or_query(text: str) -> str | None:
    """'python backend, remote' -> 'python | backend | remote' for to_tsquery."""
    words = {w.strip(".").lower() for w in _WORD.findall(text)}
    terms = [w.replace("'", "") for w in words if len(w) > 1]
    return " | ".join(f"'{t}'" for t in sorted(terms)) or None


def keyword_ids(conn: psycopg.Connection, query: str, filters: Filters, k: int = CANDIDATES) -> list[int]:
    tsq = or_query(query)
    if not tsq:
        return []
    where, params = filters.sql()
    rows = conn.execute(
        f"""select id from jobs, to_tsquery('simple', %(tsq)s) q
            where search_tsv @@ q and {where}
            order by ts_rank_cd(search_tsv, q) desc, posted_at desc nulls last
            limit %(k)s""",
        {**params, "tsq": tsq, "k": k},
    )
    return [r["id"] for r in rows]


def vector_ids(conn: psycopg.Connection, qvec: np.ndarray, filters: Filters, k: int = CANDIDATES) -> list[int]:
    where, params = filters.sql()
    # Filtered HNSW scans can come back short; iterative scans keep going until k rows pass.
    conn.execute("set hnsw.iterative_scan = relaxed_order")
    rows = conn.execute(
        f"""select id from jobs where embedding is not null and {where}
            order by embedding <=> %(q)s limit %(k)s""",
        {**params, "q": qvec, "k": k},
    )
    return [r["id"] for r in rows]


def rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))


def feedback(conn: psycopg.Connection, user_id: str) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
    rows = conn.execute(
        "select i.job_id, i.action, j.embedding from interactions i join jobs j on j.id = i.job_id where i.user_id = %s",
        (user_id,),
    ).fetchall()
    saved = [np.asarray(r["embedding"]) for r in rows if r["action"] == "saved"]
    skipped = [np.asarray(r["embedding"]) for r in rows if r["action"] == "skipped"]
    skipped_ids = [r["job_id"] for r in rows if r["action"] == "skipped"]
    return (np.mean(saved, axis=0) if saved else None, np.mean(skipped, axis=0) if skipped else None, skipped_ids)


def personalise(qvec: np.ndarray, saved: np.ndarray | None, skipped: np.ndarray | None) -> np.ndarray:
    v = qvec.copy()
    if saved is not None:
        v += SAVED_WEIGHT * saved
    if skipped is not None:
        v -= SKIPPED_WEIGHT * skipped
    return v / np.linalg.norm(v)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANK_MODEL)


def rerank(conn: psycopg.Connection, query: str, ids: list[int]) -> list[int]:
    if not ids:
        return ids
    rows = {r["id"]: r for r in conn.execute("select id, title, tags, location, description from jobs where id = any(%s)", (ids,))}
    docs = [embed.job_text(rows[i]) for i in ids]
    scores = list(_reranker().rerank(query, docs))
    return [i for _, i in sorted(zip(scores, ids), key=lambda p: -p[0])]


def search_ids(
    conn: psycopg.Connection,
    query: str,
    *,
    mode: Mode = "hybrid",
    filters: Filters | None = None,
    user_id: str | None = None,
    limit: int = 10,
    use_rerank: bool = False,
    rerank_depth: int = 30,
) -> list[int]:
    filters = filters or Filters()
    qvec = None
    if mode != "keyword":
        qvec = embed.embed_one(query)
        if user_id:
            saved, skipped, skipped_ids = feedback(conn, user_id)
            qvec = personalise(qvec, saved, skipped)
            filters.exclude_ids = [*filters.exclude_ids, *skipped_ids]

    if mode == "keyword":
        ids = keyword_ids(conn, query, filters)
    elif mode == "vector":
        ids = vector_ids(conn, qvec, filters)
    else:
        ids = rrf(keyword_ids(conn, query, filters), vector_ids(conn, qvec, filters))

    if use_rerank:
        ids = rerank(conn, query, ids[:rerank_depth]) + ids[rerank_depth:]
    return ids[:limit]


def fetch_jobs(conn: psycopg.Connection, ids: list[int], description_chars: int = 400) -> list[dict]:
    rows = {
        r["id"]: r
        for r in conn.execute(
            """select id, title, company, location, remote, tags, job_types, url, posted_at,
                      left(description, %s) as snippet
               from jobs where id = any(%s)""",
            (description_chars, ids),
        )
    }
    return [rows[i] for i in ids if i in rows]
