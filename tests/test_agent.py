"""Agent loop tests with a scripted model: no API key, deterministic."""

import json

from rolecall.agent import Agent

from .conftest import job_id


class ScriptedLLM:
    """Plays back assistant turns. A turn may be a function of the messages so far."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        turn = self.turns.pop(0)
        return turn(messages) if callable(turn) else turn


def call(name, n=0, **args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def first_id(messages) -> int:
    return json.loads(next(m["content"] for m in messages if m["role"] == "tool"))[0]["job_id"]


def submit(job, fit="strong", n=9):
    return call("submit_shortlist", n, matches=[{"job_id": job, "fit": fit, "why": "Python backend work, as asked."}])


def test_search_then_submit_is_accepted(seeded):
    llm = ScriptedLLM([call("search_jobs", query="python backend", remote=False), lambda m: submit(first_id(m))])
    result = Agent(seeded, llm).ask("python backend jobs please")
    assert result.shortlist and result.shortlist.matches[0].job_id == job_id(seeded, "py-backend")
    assert result.repairs == 0
    assert [t["tool"] for t in result.trace] == ["search_jobs", "submit_shortlist"]


def test_ungrounded_job_id_is_rejected_then_repaired(seeded):
    unseen = job_id(seeded, "sales")
    llm = ScriptedLLM([call("search_jobs", query="python backend", limit=1), submit(unseen), lambda m: submit(first_id(m), n=10)])
    result = Agent(seeded, llm).ask("python backend")
    assert result.repairs == 1
    assert "never came back" in llm.seen[2][-1]["content"]
    assert result.shortlist.matches[0].job_id != unseen


def test_schema_violation_is_rejected(seeded):
    llm = ScriptedLLM([call("search_jobs", query="python"), lambda m: submit(first_id(m), fit="perfect"),
                       lambda m: submit(first_id(m), n=10)])
    result = Agent(seeded, llm).ask("python")
    assert result.repairs == 1 and result.shortlist.matches[0].fit == "strong"


def test_gives_up_after_repeated_bad_answers(seeded):
    bad = submit(999_999)
    result = Agent(seeded, ScriptedLLM([bad] * 4)).ask("anything")
    assert result.shortlist is None and result.text == "Couldn't produce a valid shortlist."


def test_plain_text_reply_is_a_clarifying_question(seeded):
    result = Agent(seeded, ScriptedLLM([{"role": "assistant", "content": "What kind of role?"}])).ask("help")
    assert result.text == "What kind of role?" and result.shortlist is None


def test_memory_carries_into_the_next_session(seeded):
    Agent(seeded, ScriptedLLM([call("remember", fact="Doesn't speak German"), {"role": "assistant", "content": "Noted."}])).ask("I don't speak German")
    llm = ScriptedLLM([{"role": "assistant", "content": "ok"}])
    Agent(seeded, llm).ask("hi")
    assert "Doesn't speak German" in llm.seen[0][0]["content"]


def test_skip_tool_hides_job_from_later_searches(seeded):
    target = job_id(seeded, "py-backend")
    llm = ScriptedLLM([call("skip_job", job_id=target), call("search_jobs", 1, query="python backend"),
                       {"role": "assistant", "content": "done"}])
    Agent(seeded, llm).ask("not that one")
    assert target not in [j["job_id"] for j in json.loads(llm.seen[2][-1]["content"])]
