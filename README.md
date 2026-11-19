<div align="center">

# rolecall

*Roll call, but for roles.*

[![Tests](https://github.com/justhashim/rolecall/actions/workflows/test.yml/badge.svg)](https://github.com/justhashim/rolecall/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-17%20%2B%20pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

[Features](#features) • [Getting started](#getting-started) • [Usage](#usage) • [How it works](#how-it-works) • [Evals](#evals)

</div>

A job-matching agent over a live jobs index. You say what you want; it searches, remembers what matters to you, and comes back with a shortlist it has to justify.

It's small on purpose. The interesting parts are the ones that usually get skipped: keeping an index in sync with a feed that changes every hour, measuring search quality instead of eyeballing it, and not letting the model recommend a job it never retrieved.

```
$ rolecall search "data engineer airflow" --remote --limit 5
   659  Working Student Data Engineer – Python, SQL & Cloud Platforms   AgentixLake GmbH       remote
   881  Senior Data Architect (remote from Europe)                       AlmavivA de Belgique   remote
  5418  Senior Consultant (m/w/d) Data Engineering SAP BW / SAP ...      Windhoff Group         remote
     6  Principal Software Engineer, Data Lakes                          Fivetran               remote
   367  Karrierestart in Data & AI - IHK Data Analyst (m/w/d) ...        DataSmart Point GmbH   remote
```

## Features

- **Live index**: crawls a public jobs feed into Postgres every hour. Only new or edited postings get re-embedded; vanished ones are closed.
- **Hybrid search**: Postgres full-text and pgvector, fused with reciprocal rank fusion, with an optional cross-encoder rerank.
- **Personalised**: jobs you save pull results towards them; jobs you skip push them away and disappear.
- **Tool-calling agent**: works with any OpenAI-compatible endpoint (OpenRouter, Groq, OpenAI, a local vLLM).
- **Grounded answers**: recommendations are schema-validated, and a job the model never retrieved is rejected.
- **Memory**: facts you tell it ("I don't speak German") carry into every later session.
- **Evals, not vibes**: 573 graded relevance judgments, a model sweep, and an agent eval with deterministic checks.
- **Runs locally**: embeddings run on CPU, so search needs no API key at all.

## Getting started

### Prerequisites

- **[Docker](https://docs.docker.com/get-docker/)** for Postgres 17 with pgvector.
- **[uv](https://docs.astral.sh/uv/)**, which also installs Python 3.13 if you don't have it.
- **An LLM endpoint with tool calling**, only for `chat` and the agent eval. Search and ingestion don't need one.

### Install

```bash
git clone https://github.com/justhashim/rolecall.git
cd rolecall
cp .env.example .env        # add LLM_* values here if you want the agent
docker compose up -d db     # Postgres + pgvector on localhost:5544
uv sync
```

### Build the index

```bash
uv run rolecall ingest
```

The first crawl takes a few minutes: about 5,600 postings, embedded on your CPU. Later crawls only embed what changed.

> [!TIP]
> The first run downloads the embedding model (~220 MB) to `~/.cache/fastembed`. Set `FASTEMBED_CACHE_PATH` to put it somewhere else. The default avoids `/tmp`, which is often a small RAM disk.

## Usage

**Search** open jobs, optionally filtered:

```bash
uv run rolecall search "react frontend developer" --remote
uv run rolecall search "Werkstudent Softwareentwicklung" --location berlin --mode keyword
```

**Talk to the agent.** `-v` prints every tool call it makes:

```bash
uv run rolecall chat -v
```

**Check the index**: open, closed and remote counts, plus the last five ingest runs:

```bash
uv run rolecall stats
```

**Keep it fresh.** This runs the ingester in a container, once an hour:

```bash
docker compose --profile worker up -d
```

### Options

```
rolecall ingest
  --max-pages N              stop after N pages of 300 postings [default: 50, the whole feed]

rolecall search <query>
  --mode MODE                keyword | vector | hybrid               [default: hybrid]
  --remote / --no-remote     only remote, or only on-site roles     [default: both]
  --location TEXT            substring match on the location
  --user ID                  personalise with this user's saved/skipped jobs
  --limit N                  number of results                       [default: 10]
  --rerank                   cross-encoder rerank of the top 30 (slow on CPU)

rolecall chat
  --user ID                  whose memory and feedback to use        [default: me]
  -v, --verbose              print every tool call
```

### Configuration

| Variable | Used by | Default |
|---|---|---|
| `DATABASE_URL` | everything | `postgresql://rolecall:rolecall@localhost:5544/rolecall` |
| `LLM_BASE_URL` | `chat`, agent eval | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | `chat`, agent eval | |
| `LLM_MODEL` | `chat`, agent eval | |
| `FASTEMBED_CACHE_PATH` | embeddings | `~/.cache/fastembed` |

## How it works

```mermaid
flowchart LR
  feed[Arbeitnow feed<br/>hourly] -->|crawl, hash, upsert| pg[(Postgres<br/>tsvector + pgvector)]
  user([you]) <--> agent[agent]
  agent <-->|tool calls| llm[LLM]
  agent -->|search_jobs / get_job| search[hybrid search]
  search --> pg
  agent -->|remember, save, skip| pg
  agent -->|submit_shortlist| check{schema +<br/>grounding}
  check -->|rejected: reason| agent
  check -->|accepted| user
```

**Ingestion** (`rolecall/ingest.py`). Every posting gets a hash of the fields that go into its embedding. An unchanged hash means only `last_seen_at` is bumped, with no re-embedding. Postings that disappear are closed, but only after a crawl that reached the last page: a partial crawl can't tell "gone" from "not fetched yet". The feed shifts while you paginate, so duplicates across pages are collapsed, and `429`s are retried using `Retry-After`. On a real re-crawl an hour after the first: 5,624 fetched, 2,560 unchanged, 102 edited and re-embedded, 1 closed.

**Search** (`rolecall/search.py`). Keyword search ORs the query terms over a weighted `tsvector` (title > tags > description) using the `simple` config, because the feed mixes German and English and a stemmer for one mangles the other. Vector search uses an HNSW index with iterative scans, so filtered queries still return enough rows. Hybrid fuses the two by rank, not score, so neither scale has to be calibrated.

**Agent** (`rolecall/agent.py`). Tools: `search_jobs`, `get_job`, `remember`, `save_job`, `skip_job` and `submit_shortlist`. The only way to recommend is `submit_shortlist`, and the call has to pass two checks:

1. **Schema**: Pydantic validates it. Fit is `strong`, `good` or `stretch`, every match has a reason, and there are at most five matches.
2. **Grounding**: every `job_id` must have come back from a search or a read in this conversation.

A failed check goes back to the model as the tool result, with the reason, and it gets two tries to repair it.

## Evals

The live index changes every hour, so evals run on a frozen snapshot of 5,624 postings (`evals/data/snapshot.jsonl.gz`), loaded into its own database.

**Retrieval** (`python -m evals.retrieval_eval run`). The eval uses 20 queries a job seeker would actually type, in English and German. Every system's top 10 was pooled and graded on a three-point scale: 2 for what was asked, 1 for related, 0 for wrong. That gives 573 judgments in `evals/data/qrels.jsonl`.

| System | nDCG@10 | Recall@10 | MRR | p50 latency |
|---|---|---|---|---|
| keyword | 0.666 | 0.395 | 0.741 | 6 ms |
| vector, English model (`bge-small-en-v1.5`) | 0.672 | 0.443 | 0.759 | 35 ms |
| vector, multilingual model (`paraphrase-multilingual-MiniLM`) | 0.468 | 0.323 | 0.506 | 17 ms |
| hybrid (keyword + multilingual vector) | 0.668 | 0.413 | 0.778 | 23 ms |
| hybrid + rerank (`bge-reranker-v2-m3`) | **0.748** | **0.458** | **0.833** | 9.4 s |

What the numbers say:

- **The first embedding model was the wrong call.** Multilingual sounded right for a German and English feed, but this model is trained for paraphrase, not retrieval, and it scored worst of all, below plain keyword search.
- **Fusion can't rescue a weak retriever.** Hybrid landed level with keyword. On `software engineering internship` it scored 0.0: the two retrievers each found real internships, disagreed, and fusion promoted generic roles sitting mid-table in both lists.
- **Reranking works, but it's too slow on CPU**, at +0.08 nDCG for about 9 seconds a query. So it stays opt-in.

> [!NOTE]
> `python -m evals.model_sweep` compares embedding models on the same labels, alone and inside hybrid. The default model will change to whichever wins, and this table will be rerun. Pooling has a known blind spot: for `Buchhaltung`, no system found the ~20 English-titled *Accountant* postings, so those were added by a direct title search.

**Agent** (`python -m evals.agent_eval`, needs `LLM_*`). Seven scenarios against a real model, each checked deterministically. Did it pass `remote=true` when asked, and invent no filters when not? Did the shortlist pass on the first try? Did it store a deal-breaker in memory? Did it ask before searching on a vague request? Recommended jobs are scored against the same labels as the retrieval eval. Results go to `evals/results/agent.json`.

**Tests** (`uv run pytest`). There are 19 tests: ingestion edge cases, search modes and filters, and the agent loop driven by a scripted fake LLM, so they need no API key. CI runs them against a real pgvector container.

## Project layout

```
db/schema.sql           tables, HNSW + GIN indexes, generated tsvector
rolecall/ingest.py      crawl, hash, upsert, re-embed changed, close vanished
rolecall/search.py      keyword, vector, RRF hybrid, rerank, personalisation
rolecall/agent.py       tool loop, memory, validated and grounded output
rolecall/llm.py         OpenAI-compatible client with retries
rolecall/cli.py         the rolecall command
evals/                  snapshot, labels, retrieval eval, model sweep, agent eval
tests/                  ingest, search, agent
```

Job data comes from the free [Arbeitnow job board API](https://www.arbeitnow.com/api/job-board-api). The snapshot keeps the first 1,500 characters of each description; full postings are at their URLs.
