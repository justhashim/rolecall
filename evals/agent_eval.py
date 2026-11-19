"""Agent eval: run a real model through fixed scenarios and check what it did, not how it sounded.

    python -m evals.agent_eval            # needs LLM_* in .env; runs against the frozen snapshot

Every check is deterministic: which tools were called, with which arguments, whether the
final shortlist passed schema and grounding on the first try, and whether the recommended
jobs are relevant according to the same labels the retrieval eval uses (where a scenario
maps to a labelled query). Results go to evals/results/agent.json.
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import Callable

from rolecall.agent import Agent, Result
from rolecall.llm import LLM

from . import snapshot
from .retrieval_eval import RESULTS, qrels


@dataclass
class Scenario:
    name: str
    turns: list[str]
    checks: dict[str, Callable[[Result, "Ctx"], bool]] = field(default_factory=dict)
    qid: str | None = None  # labelled query this scenario corresponds to, for precision of the shortlist


@dataclass
class Ctx:
    conn: object
    user: str


def called(result: Result, tool: str, **expect) -> bool:
    for t in result.trace:
        if t["tool"] != tool or not t["ok"]:
            continue
        args = json.loads(t["args"] or "{}")
        if all(args.get(k) == v or (isinstance(v, str) and v.lower() in str(args.get(k, "")).lower()) for k, v in expect.items()):
            return True
    return False


def has_shortlist(r: Result, _ctx) -> bool:
    return r.shortlist is not None


def first_try(r: Result, _ctx) -> bool:
    return r.shortlist is not None and r.repairs == 0


def memory_has(word: str):
    return lambda _r, ctx: any(word in f.lower() for f in [m["fact"] for m in ctx.conn.execute("select fact from memories where user_id = %s", (ctx.user,))])


SCENARIOS = [
    Scenario("plain search", ["I'm a junior Python backend developer. What should I apply to?"], qid="q01",
             checks={"searched": lambda r, _: called(r, "search_jobs"), "shortlist": has_shortlist, "valid first try": first_try}),
    Scenario("remote filter", ["Only fully remote roles please. I do React frontend work."], qid="q02",
             checks={"remote=true": lambda r, _: called(r, "search_jobs", remote=True), "shortlist": has_shortlist}),
    Scenario("location filter", ["Data engineer jobs in Berlin?"], qid="q16",
             checks={"location=berlin": lambda r, _: called(r, "search_jobs", location="berlin"), "shortlist": has_shortlist}),
    Scenario("no invented filters", ["Machine learning roles where I'd work on LLMs."], qid="q03",
             checks={"no remote filter": lambda r, _: not called(r, "search_jobs", remote=True) and not called(r, "search_jobs", remote=False),
                     "shortlist": has_shortlist}),
    Scenario("cross-language", ["Ich suche eine Stelle als Buchhalter."], qid="q13",
             checks={"searched": lambda r, _: called(r, "search_jobs"), "shortlist": has_shortlist}),
    Scenario("remembers deal-breakers", ["Remember this: I don't speak German, so skip German-only roles.", "Find me customer support jobs."], qid="q07",
             checks={"remember called": lambda r, ctx: memory_has("german")(r, ctx), "shortlist": has_shortlist}),
    Scenario("vague request asks first", ["Find me a job."],
             checks={"asked a question": lambda r, _: r.shortlist is None and bool(r.text) and "?" in r.text}),
]


def run() -> None:
    conn = snapshot.load()
    labels = qrels()
    llm = LLM.from_env()
    rows = []
    for i, sc in enumerate(SCENARIOS):
        user = f"eval-{i}"
        conn.execute("delete from users where id = %s", (user,))
        agent = Agent(conn, llm, user_id=user)
        calls_before = len(llm.calls)
        result = Result()
        for turn in sc.turns:
            result = agent.ask(turn)
        ctx = Ctx(conn, user)
        checks = {name: bool(fn(result, ctx)) for name, fn in sc.checks.items()}
        precision = None
        if sc.qid and result.shortlist:
            ids = [m.job_id for m in result.shortlist.matches]
            sids = {r["id"]: r["source_id"] for r in conn.execute("select id, source_id from jobs where id = any(%s)", (ids,))}
            got = labels.get(sc.qid, {})
            precision = statistics.mean(1.0 if got.get(sids[i], 0) >= 1 else 0.0 for i in ids)
        calls = llm.calls[calls_before:]
        rows.append({"scenario": sc.name, "checks": checks, "passed": all(checks.values()), "repairs": result.repairs,
                     "shortlist_precision": precision, "llm_calls": len(calls),
                     "latency_s": round(sum(c["ms"] for c in calls) / 1000, 1), "tools": [t["tool"] for t in result.trace]})
        mark = "PASS" if rows[-1]["passed"] else "FAIL"
        print(f"{mark}  {sc.name:<26} {checks}  precision={precision}  repairs={result.repairs}  {rows[-1]['latency_s']}s")

    precisions = [r["shortlist_precision"] for r in rows if r["shortlist_precision"] is not None]
    summary = {
        "model": llm.model,
        "scenarios_passed": f"{sum(r['passed'] for r in rows)}/{len(rows)}",
        "checks_passed": f"{sum(sum(r['checks'].values()) for r in rows)}/{sum(len(r['checks']) for r in rows)}",
        "mean_shortlist_precision": round(statistics.mean(precisions), 3) if precisions else None,
        "total_repairs": sum(r["repairs"] for r in rows),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "agent.json").write_text(json.dumps({"summary": summary, "scenarios": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
