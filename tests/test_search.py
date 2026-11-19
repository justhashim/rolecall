from rolecall import db, search

from .conftest import job_id


def slugs(conn, ids):
    rows = {r["id"]: r["source_id"] for r in conn.execute("select id, source_id from jobs where id = any(%s)", (ids,))}
    return [rows[i] for i in ids]


def test_or_query_tolerates_punctuation_and_quotes():
    assert search.or_query("C++, node.js & O'Reilly") == "'c++' | 'node.js' | 'oreilly'"
    assert search.or_query("!!") is None


def test_rrf_rewards_agreement():
    assert search.rrf([1, 2, 3], [3, 1, 4]) == [1, 3, 2, 4]


def test_each_mode_finds_the_obvious_job(seeded):
    for mode in ("keyword", "vector", "hybrid"):
        assert slugs(seeded, search.search_ids(seeded, "python backend developer", mode=mode, limit=1)) == ["py-backend"]


def test_vector_search_crosses_languages(seeded):
    # The posting is German only; the query is English. Keyword search can't match it.
    assert "nurse" not in slugs(seeded, search.search_ids(seeded, "nurse hospital", mode="keyword"))
    assert slugs(seeded, search.search_ids(seeded, "nurse hospital", mode="vector", limit=1)) == ["nurse"]


def test_filters(seeded):
    remote = slugs(seeded, search.search_ids(seeded, "engineer", filters=search.Filters(remote=True)))
    assert set(remote) == {"react-fe", "data-eng"}
    assert slugs(seeded, search.search_ids(seeded, "job", filters=search.Filters(location="hamburg"))) == ["nurse"]


def test_skipped_jobs_are_hidden_and_saved_jobs_pull_results(seeded):
    db.ensure_user(seeded, "u")
    seeded.execute("insert into interactions values ('u', %s, 'skipped')", (job_id(seeded, "py-backend"),))
    seeded.execute("insert into interactions values ('u', %s, 'saved')", (job_id(seeded, "data-eng"),))
    results = slugs(seeded, search.search_ids(seeded, "python", user_id="u"))
    assert "py-backend" not in results
    assert results[0] == "data-eng"

