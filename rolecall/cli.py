import argparse
import json
import sys

from . import db, ingest, search


def cmd_ingest(args) -> None:
    conn = db.connect()
    db.migrate(conn)
    raw, complete = ingest.fetch(max_pages=args.max_pages)
    stats = ingest.sync(conn, raw, complete)
    print(json.dumps({**stats.__dict__, "complete_crawl": complete}))


def cmd_search(args) -> None:
    conn = db.connect()
    filters = search.Filters(remote=args.remote, location=args.location)
    ids = search.search_ids(conn, args.query, mode=args.mode, filters=filters, user_id=args.user, limit=args.limit, use_rerank=args.rerank)
    for job in search.fetch_jobs(conn, ids):
        where = "remote" if job["remote"] else job["location"]
        print(f"{job['id']:>6}  {job['title'][:70]:<70}  {job['company'][:28]:<28}  {where}")


def cmd_chat(args) -> None:
    from .agent import Agent
    from .llm import LLM

    conn = db.connect()
    db.migrate(conn)
    agent = Agent(conn, LLM.from_env(), user_id=args.user, verbose=args.verbose)
    print("rolecall. Tell it what you're looking for. Ctrl-D to quit.")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        if not line.strip():
            continue
        result = agent.ask(line.strip())
        print(result.render(conn))


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
    s.add_argument("--max-pages", type=int, default=50, help="stop after this many pages (300 postings each); default reaches the end of the feed")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="search open jobs")
    s.add_argument("query")
    s.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    s.add_argument("--remote", action=argparse.BooleanOptionalAction, default=None)
    s.add_argument("--location")
    s.add_argument("--user")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--rerank", action="store_true")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("chat", help="talk to the agent")
    s.add_argument("--user", default="me")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("stats", help="index size and recent ingest runs")
    s.set_defaults(fn=cmd_stats)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
