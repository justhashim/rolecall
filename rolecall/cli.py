import argparse
import json

from . import db, ingest


def cmd_ingest(args) -> None:
    conn = db.connect()
    db.migrate(conn)
    raw, complete = ingest.fetch(max_pages=args.max_pages)
    stats = ingest.sync(conn, raw, complete)
    print(json.dumps({**stats.__dict__, "complete_crawl": complete}))


def cmd_stats(_args) -> None:
    conn = db.connect()
    row = conn.execute(
        """select count(*) filter (where closed_at is null) as open, count(*) filter (where closed_at is not null) as closed,
                  count(*) filter (where remote and closed_at is null) as remote_open from jobs"""
    ).fetchone()
    runs = conn.execute("select * from ingest_runs order by id desc limit 5").fetchall()
    print(json.dumps({**row, "recent_runs": runs}, default=str, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="rolecall")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("ingest", help="crawl the feed and sync the jobs table")
    s.add_argument("--max-pages", type=int, default=10)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("stats", help="index size and recent ingest runs")
    s.set_defaults(fn=cmd_stats)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
