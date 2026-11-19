from rolecall import ingest

from .conftest import FEED, raw


def test_strip_html_keeps_line_breaks_and_entities():
    assert ingest.strip_html("<p>Hello&nbsp;<b>world</b></p><ul><li>one</li><li>two &amp; three</li></ul>") == "Hello\xa0world\none\ntwo & three"


def test_resync_of_same_feed_embeds_nothing(conn):
    first = ingest.sync(conn, FEED, complete=True)
    second = ingest.sync(conn, FEED, complete=True)
    assert (first.inserted, first.unchanged) == (5, 0)
    assert (second.inserted, second.updated, second.unchanged, second.closed) == (0, 0, 5, 0)


def test_edited_posting_is_updated_and_reembedded(conn):
    ingest.sync(conn, FEED, complete=True)
    before = conn.execute("select embedding from jobs where source_id = 'sales'").fetchone()["embedding"]
    edited = [*FEED[:-1], raw("sales", "Account Executive SaaS", tags=["Sales"], location="Munich", description="Now fully remote, selling to banks.")]
    stats = ingest.sync(conn, edited, complete=True)
    after = conn.execute("select embedding from jobs where source_id = 'sales'").fetchone()["embedding"]
    assert (stats.updated, stats.unchanged) == (1, 4)
    assert before != after


def test_missing_posting_closes_only_after_a_complete_crawl(conn):
    ingest.sync(conn, FEED, complete=True)
    assert ingest.sync(conn, FEED[1:], complete=False).closed == 0
    assert ingest.sync(conn, FEED[1:], complete=True).closed == 1
    back = ingest.sync(conn, FEED, complete=True)
    assert back.reopened == 1
    assert conn.execute("select count(*) as n from jobs where closed_at is null").fetchone()["n"] == 5


def test_duplicate_slugs_across_pages_count_once(conn):
    stats = ingest.sync(conn, FEED + FEED[:2], complete=True)
    assert stats.fetched == 5
