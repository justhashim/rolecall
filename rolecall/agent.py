"""The job-search agent: tools, memory, and an answer that has to pass validation.

The model never answers in free text when it has jobs to recommend. It calls
`submit_shortlist`, and that call is checked twice before it's accepted:

1. Schema: Pydantic validates the arguments (fit level, reasons, at most 5 matches).
2. Grounding: every job_id must have come back from a search_jobs or get_job call in
   this conversation. A model can't recommend a job it never retrieved.

A failed check goes back to the model as the tool result, with the reason, and it gets
MAX_REPAIRS tries to fix it. Memory has two layers: the message history within a
session, and `memories` rows (written by the `remember` tool) loaded into every session.
"""

import json
from dataclasses import dataclass, field
from typing import Literal

import psycopg
from pydantic import BaseModel, Field, ValidationError

from . import db, search
from .llm import LLM

MAX_STEPS = 10
MAX_REPAIRS = 2


class Match(BaseModel):
    job_id: int
    fit: Literal["strong", "good", "stretch"]
    why: str = Field(min_length=10, max_length=300, description="Why this job fits, citing the user's own words or facts.")
    gaps: list[str] = Field(default_factory=list, max_length=3, description="What the user would be missing, if anything.")


class Shortlist(BaseModel):
    matches: list[Match] = Field(min_length=1, max_length=5)
    note: str | None = Field(default=None, max_length=300, description="One line for the user, e.g. a follow-up question.")


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _tool("search_jobs", "Search open jobs. Write the query as a short description of the role, in any language.",
          {"query": {"type": "string"},
           "remote": {"type": "boolean", "description": "true = remote only, false = on-site only. Omit if the user doesn't care."},
           "location": {"type": "string", "description": "City or country to filter on. Omit for anywhere."},
           "limit": {"type": "integer", "minimum": 1, "maximum": 15}},
          ["query"]),
    _tool("get_job", "Read one job's full posting.", {"job_id": {"type": "integer"}}, ["job_id"]),
    _tool("remember", "Store a lasting fact or preference about the user (skills, location, deal-breakers).",
          {"fact": {"type": "string"}}, ["fact"]),
    _tool("save_job", "The user liked this job. Future searches lean towards similar ones.", {"job_id": {"type": "integer"}}, ["job_id"]),
    _tool("skip_job", "The user rejected this job. Future searches lean away and hide it.", {"job_id": {"type": "integer"}}, ["job_id"]),
    _tool("submit_shortlist", "Give the user your recommendations. The only way to recommend jobs.",
          Shortlist.model_json_schema()["properties"] | {}, ["matches"]),
]
# Pydantic puts Match in $defs; inline it so models that ignore $ref still see the shape.
TOOLS[-1]["function"]["parameters"]["properties"]["matches"]["items"] = Match.model_json_schema()

SYSTEM = """You help one person find jobs from a live index of postings (mostly Germany and Europe, many in German).

How to work:
- Search before recommending. Run more than one search when the request has several angles.
- Use filters only for what the user actually said (remote, a city). Don't invent constraints.
- Call `remember` when the user tells you something that should still matter next session.
- Call save_job / skip_job when the user reacts to a specific job.
- To recommend, call submit_shortlist. Only use job_ids you got from search_jobs or get_job.
  Be honest about fit: "stretch" with gaps is more useful than a flattering "strong".
- If the request is too vague to search, ask one short question instead (plain text, no tool).

What you know about this person:
{memories}"""


@dataclass
class Result:
    shortlist: Shortlist | None = None
    text: str | None = None
    trace: list[dict] = field(default_factory=list)  # every tool call: name, args, ok
    repairs: int = 0

    def render(self, conn: psycopg.Connection) -> str:
        if not self.shortlist:
            return self.text or "(no answer)"
        jobs = {j["id"]: j for j in search.fetch_jobs(conn, [m.job_id for m in self.shortlist.matches])}
        lines = []
        for m in self.shortlist.matches:
            j = jobs[m.job_id]
            lines.append(f"[{m.fit}] {j['title']} at {j['company']} ({'remote' if j['remote'] else j['location']})")
            lines.append(f"  why: {m.why}")
            lines += [f"  gap: {g}" for g in m.gaps]
            lines.append(f"  {j['url']}")
        if self.shortlist.note:
            lines.append(self.shortlist.note)
        return "\n".join(lines)


class Agent:
    def __init__(self, conn: psycopg.Connection, llm: LLM, user_id: str = "me", verbose: bool = False):
        self.conn, self.llm, self.user_id, self.verbose = conn, llm, user_id, verbose
        db.ensure_user(conn, user_id)
        self.messages: list[dict] = [{"role": "system", "content": ""}]
        self.seen_ids: set[int] = set()

    def memories(self) -> list[str]:
        rows = self.conn.execute("select fact from memories where user_id = %s order by id", (self.user_id,))
        return [r["fact"] for r in rows]

    def ask(self, text: str) -> Result:
        facts = self.memories()
        self.messages[0]["content"] = SYSTEM.format(memories="\n".join(f"- {f}" for f in facts) or "- nothing yet")
        self.messages.append({"role": "user", "content": text})
        result = Result()

        for _ in range(MAX_STEPS):
            reply = self.llm.chat(self.messages, TOOLS)
            self.messages.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")})
            calls = reply.get("tool_calls") or []
            if not calls:
                result.text = reply.get("content") or ""
                return result
            for call in calls:
                name = call["function"]["name"]
                output, ok, accepted = self._run(name, call["function"].get("arguments") or "{}", result)
                result.trace.append({"tool": name, "args": call["function"].get("arguments"), "ok": ok})
                if self.verbose:
                    print(f"  · {name} {call['function'].get('arguments')} -> {'ok' if ok else output[:120]}")
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                if accepted:
                    result.shortlist = accepted
            if result.shortlist:
                return result
            if result.repairs > MAX_REPAIRS:
                result.text = "Couldn't produce a valid shortlist."
                return result
        result.text = "Ran out of steps."
        return result

    def _run(self, name: str, raw_args: str, result: Result) -> tuple[str, bool, Shortlist | None]:
        """Returns (tool output for the model, whether the call succeeded, accepted shortlist if any)."""
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            result.repairs += name == "submit_shortlist"
            return f"error: arguments are not valid JSON ({e})", False, None

        if name == "submit_shortlist":
            try:
                shortlist = Shortlist.model_validate(args)
            except ValidationError as e:
                result.repairs += 1
                return f"rejected: {e.errors(include_url=False)}. Fix and call submit_shortlist again.", False, None
            unknown = [m.job_id for m in shortlist.matches if m.job_id not in self.seen_ids]
            dupes = len({m.job_id for m in shortlist.matches}) != len(shortlist.matches)
            if unknown or dupes:
                result.repairs += 1
                why = f"job_ids {unknown} never came back from search_jobs/get_job" if unknown else "duplicate job_ids"
                return f"rejected: {why}. Only recommend jobs you retrieved.", False, None
            return "accepted", True, shortlist

        try:
            return getattr(self, f"_tool_{name}")(**args), True, None
        except (AttributeError, TypeError) as e:
            return f"error: {e}", False, None

    def _tool_search_jobs(self, query: str, remote: bool | None = None, location: str | None = None, limit: int = 8) -> str:
        filters = search.Filters(remote=remote, location=location)
        ids = search.search_ids(self.conn, query, filters=filters, user_id=self.user_id, limit=max(1, min(limit, 15)))
        jobs = search.fetch_jobs(self.conn, ids, description_chars=300)
        self.seen_ids.update(j["id"] for j in jobs)
        return json.dumps([
            {"job_id": j["id"], "title": j["title"], "company": j["company"],
             "where": "remote" if j["remote"] else j["location"], "tags": j["tags"][:5], "snippet": j["snippet"]}
            for j in jobs
        ], ensure_ascii=False) if jobs else "no results; try broader words or fewer filters"

    def _tool_get_job(self, job_id: int) -> str:
        row = self.conn.execute(
            "select id, title, company, location, remote, tags, job_types, url, left(description, 3000) as description "
            "from jobs where id = %s and closed_at is null", (job_id,)
        ).fetchone()
        if not row:
            return f"error: no open job {job_id}"
        self.seen_ids.add(job_id)
        return json.dumps(row, ensure_ascii=False, default=str)

    def _tool_remember(self, fact: str) -> str:
        self.conn.execute("insert into memories (user_id, fact) values (%s, %s) on conflict do nothing", (self.user_id, fact.strip()))
        return "remembered"

    def _feedback(self, job_id: int, action: str) -> str:
        if not self.conn.execute("select 1 from jobs where id = %s", (job_id,)).fetchone():
            return f"error: no job {job_id}"
        self.conn.execute(
            """insert into interactions (user_id, job_id, action) values (%s, %s, %s)
               on conflict (user_id, job_id) do update set action = excluded.action, created_at = now()""",
            (self.user_id, job_id, action),
        )
        return action

    def _tool_save_job(self, job_id: int) -> str:
        return self._feedback(job_id, "saved")

    def _tool_skip_job(self, job_id: int) -> str:
        return self._feedback(job_id, "skipped")
