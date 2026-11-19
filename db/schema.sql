-- Idempotent: safe to run on every start.
create extension if not exists vector;

-- array_to_string is only STABLE, and a generated column needs IMMUTABLE.
-- For text[] it really is immutable, so wrap it.
create or replace function tags_text(text[]) returns text
  language sql immutable parallel safe
  as $$ select coalesce(array_to_string($1, ' '), '') $$;

create table if not exists jobs (
  id             bigserial primary key,
  source         text        not null,
  source_id      text        not null,
  title          text        not null,
  company        text        not null,
  location       text        not null default '',
  remote         boolean     not null default false,
  tags           text[]      not null default '{}',
  job_types      text[]      not null default '{}',
  description    text        not null default '',
  url            text        not null,
  posted_at      timestamptz,
  -- sha256 of the fields that go into the embedding. Unchanged hash = skip re-embedding.
  content_hash   text        not null,
  embedding      vector(384),
  -- 'simple' config on purpose: the feed is mostly German with some English,
  -- and a language stemmer would mangle the other half.
  search_tsv     tsvector generated always as (
                   setweight(to_tsvector('simple', title), 'A') ||
                   setweight(to_tsvector('simple', tags_text(tags)), 'B') ||
                   setweight(to_tsvector('simple', description), 'C')
                 ) stored,
  first_seen_at  timestamptz not null default now(),
  last_seen_at   timestamptz not null default now(),
  closed_at      timestamptz,
  unique (source, source_id)
);

create index if not exists jobs_embedding_hnsw on jobs using hnsw (embedding vector_cosine_ops);
create index if not exists jobs_search_tsv on jobs using gin (search_tsv);
create index if not exists jobs_open_recent on jobs (posted_at desc) where closed_at is null;

create table if not exists users (
  id          text primary key,
  created_at  timestamptz not null default now()
);

-- Long-term memory the agent writes with the `remember` tool.
create table if not exists memories (
  id          bigserial primary key,
  user_id     text not null references users(id) on delete cascade,
  fact        text not null,
  created_at  timestamptz not null default now(),
  unique (user_id, fact)
);

-- Feedback that personalises search: saved jobs pull the query vector towards them,
-- skipped jobs push it away and are hidden.
create table if not exists interactions (
  user_id     text   not null references users(id) on delete cascade,
  job_id      bigint not null references jobs(id) on delete cascade,
  action      text   not null check (action in ('saved', 'skipped')),
  created_at  timestamptz not null default now(),
  primary key (user_id, job_id)
);

create table if not exists ingest_runs (
  id           bigserial primary key,
  source       text        not null,
  started_at   timestamptz not null,
  finished_at  timestamptz,
  complete     boolean     not null default false,
  fetched      int not null default 0,
  inserted     int not null default 0,
  updated      int not null default 0,
  unchanged    int not null default 0,
  reopened     int not null default 0,
  closed       int not null default 0
);
