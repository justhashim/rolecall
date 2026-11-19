"""Freeze the live index into a file, and load that file into its own database.

The live table changes every hour, so evals run against a frozen copy in a separate
database (`rolecall_eval`). Labels in qrels.jsonl are keyed by source_id.
"""

import gzip
import json
import sys
from pathlib import Path

import psycopg

from rolecall import db, embed

DATA = Path(__file__).parent / "data"
SNAPSHOT = DATA / "snapshot.jsonl.gz"
EVAL_DB = "rolecall_eval"
DESCRIPTION_CHARS = 1500  # enough for search; full postings live at their url

FIELDS = ["source_id", "title", "company", "location", "remote", "tags", "job_types", "description", "url", "posted_at"]


def export() -> None:
    conn = db.connect()
    rows = conn.execute(f"select {', '.join(FIELDS)} from jobs where closed_at is null order by source_id").fetchall()
    DATA.mkdir(exist_ok=True)
    with gzip.open(SNAPSHOT, "wt", encoding="utf-8") as f:
        for r in rows:
            r["description"] = r["description"][:DESCRIPTION_CHARS]
            r["posted_at"] = r["posted_at"].isoformat() if r["posted_at"] else None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} jobs to {SNAPSHOT}")


def read() -> list[dict]:
    with gzip.open(SNAPSHOT, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_url() -> str:
    info = db.connect().info
    return f"postgresql://{info.user}:{info.password}@{info.host}:{info.port}/{EVAL_DB}"


def load() -> psycopg.Connection:
    """Create and fill the eval database if it isn't there yet. Returns a connection to it."""
    admin = db.connect()
    if not admin.execute("select 1 from pg_database where datname = %s", (EVAL_DB,)).fetchone():
        admin.execute(f"create database {EVAL_DB}")
    conn = db.connect(eval_url())
    db.migrate(conn)
    if conn.execute("select count(*) as n from jobs").fetchone()["n"] == 0:
        jobs = read()
        print(f"loading {len(jobs)} snapshot jobs into {EVAL_DB} (embedding once)...", file=sys.stderr)
        vectors = embed.embed([embed.job_text(j) for j in jobs])
        with conn.transaction():
            conn.cursor().executemany(
                """insert into jobs (source, source_id, title, company, location, remote, tags, job_types,
                                     description, url, posted_at, content_hash, embedding)
                   values ('snapshot', %(source_id)s, %(title)s, %(company)s, %(location)s, %(remote)s, %(tags)s,
                           %(job_types)s, %(description)s, %(url)s, %(posted_at)s, '', %(embedding)s)""",
                [{**j, "embedding": v} for j, v in zip(jobs, vectors)],
            )
    return conn


if __name__ == "__main__":
    {"export": export, "load": load}[sys.argv[1]]()
