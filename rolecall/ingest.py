"""Keep the jobs table in sync with a feed that changes every hour.

New postings are inserted and embedded. Edited postings are re-embedded. Postings whose
content hash hasn't changed only get `last_seen_at` bumped, so a re-crawl of the same
feed embeds nothing. A posting that disappears is closed, but only after a crawl that
reached the last page: a partial crawl can't tell "gone" from "not fetched yet".
"""

import hashlib
import html
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import httpx
import psycopg

from . import embed

SOURCE = "arbeitnow"
FEED_URL = "https://www.arbeitnow.com/api/job-board-api"

_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(p|div|li|ul|ol|h\d|br)\b[^>]*>", re.I)
_SPACE = re.compile(r"[ \t\r\f\v]+")


def strip_html(raw: str) -> str:
    text = _BLOCK.sub("\n", raw or "")
    text = html.unescape(_TAG.sub("", text))
    lines = (_SPACE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def normalise(raw: dict) -> dict:
    job = {
        "source_id": raw["slug"],
        "title": raw["title"].strip(),
        "company": raw["company_name"].strip(),
        "location": (raw.get("location") or "").strip(),
        "remote": bool(raw.get("remote")),
        "tags": sorted({t.strip() for t in raw.get("tags") or [] if t.strip()}),
        "job_types": sorted({t.strip() for t in raw.get("job_types") or [] if t.strip()}),
        "description": strip_html(raw.get("description", "")),
        "url": raw["url"],
        "posted_at": datetime.fromtimestamp(raw["created_at"], UTC) if raw.get("created_at") else None,
    }
    job["content_hash"] = content_hash(job)
    return job


def content_hash(job: dict) -> str:
    fields = [job["title"], job["company"], job["location"], str(job["remote"]), "|".join(job["tags"]), job["description"]]
    return hashlib.sha256("\x1f".join(fields).encode()).hexdigest()


def fetch(max_pages: int = 10, delay: float = 1.0) -> tuple[list[dict], bool]:
    """Returns (raw postings, whether the crawl reached the last page)."""
    jobs, url, pages = [], FEED_URL, 0
    with httpx.Client(timeout=30, headers={"User-Agent": "rolecall-ingest"}) as client:
        while url and pages < max_pages:
            resp = client.get(url)
            resp.raise_for_status()
            body = resp.json()
            jobs.extend(body["data"])
            url = (body.get("links") or {}).get("next")
            pages += 1
            if url:
                time.sleep(delay)  # the API asks not to be hammered
    return jobs, url is None


@dataclass
class RunStats:
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    reopened: int = 0
    closed: int = 0


def sync(conn: psycopg.Connection, raw_jobs: list[dict], complete: bool, source: str = SOURCE) -> RunStats:
    started = conn.execute("select now() as t").fetchone()["t"]
    # The feed shifts while we paginate, so the same posting can show up on two pages.
    seen = {j["source_id"]: j for j in map(normalise, raw_jobs)}
    stats = RunStats(fetched=len(seen))

    existing = {
        r["source_id"]: r
        for r in conn.execute(
            "select source_id, content_hash, closed_at from jobs where source = %s and source_id = any(%s)",
            (source, list(seen)),
        )
    }
    new = [j for sid, j in seen.items() if sid not in existing]
    changed = [j for sid, j in seen.items() if sid in existing and existing[sid]["content_hash"] != j["content_hash"]]
    same = [sid for sid in seen if sid in existing and existing[sid]["content_hash"] == seen[sid]["content_hash"]]
    stats.reopened = sum(1 for sid in seen if sid in existing and existing[sid]["closed_at"] is not None)

    to_embed = new + changed
    vectors = embed.embed([embed.job_text(j) for j in to_embed])
    for job, vec in zip(to_embed, vectors):
        job["embedding"] = vec

    with conn.transaction():
        conn.cursor().executemany(
            """
            insert into jobs (source, source_id, title, company, location, remote, tags, job_types,
                              description, url, posted_at, content_hash, embedding)
            values (%(source)s, %(source_id)s, %(title)s, %(company)s, %(location)s, %(remote)s, %(tags)s,
                    %(job_types)s, %(description)s, %(url)s, %(posted_at)s, %(content_hash)s, %(embedding)s)
            on conflict (source, source_id) do update set
              title = excluded.title, company = excluded.company, location = excluded.location,
              remote = excluded.remote, tags = excluded.tags, job_types = excluded.job_types,
              description = excluded.description, url = excluded.url, posted_at = excluded.posted_at,
              content_hash = excluded.content_hash, embedding = excluded.embedding,
              last_seen_at = now(), closed_at = null
            """,
            [{**j, "source": source} for j in to_embed],
        )
        conn.execute(
            "update jobs set last_seen_at = now(), closed_at = null where source = %s and source_id = any(%s)",
            (source, same),
        )
        if complete:
            stats.closed = conn.execute(
                "update jobs set closed_at = now() where source = %s and closed_at is null and last_seen_at < %s",
                (source, started),
            ).rowcount
        stats.inserted, stats.updated, stats.unchanged = len(new), len(changed), len(same)
        conn.execute(
            """insert into ingest_runs (source, started_at, finished_at, complete, fetched, inserted,
                                        updated, unchanged, reopened, closed)
               values (%(source)s, %(started)s, now(), %(complete)s, %(fetched)s, %(inserted)s,
                       %(updated)s, %(unchanged)s, %(reopened)s, %(closed)s)""",
            {**asdict(stats), "source": source, "started": started, "complete": complete},
        )
    return stats
