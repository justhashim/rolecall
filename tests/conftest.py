import pytest

from rolecall import db, ingest

TEST_DB = "rolecall_test"


def raw(slug: str, title: str, *, tags=(), remote=False, location="Berlin", description="", created_at=1763500000) -> dict:
    """A posting shaped like the Arbeitnow API returns it."""
    return {"slug": slug, "company_name": "Acme", "title": title, "description": f"<p>{description or title}</p>",
            "remote": remote, "url": f"https://example.com/{slug}", "tags": list(tags), "job_types": [],
            "location": location, "created_at": created_at}


FEED = [
    raw("py-backend", "Junior Python Backend Developer", tags=["Python", "Django"], description="Build APIs in Python and PostgreSQL."),
    raw("react-fe", "Frontend Engineer (React)", tags=["React", "TypeScript"], remote=True, description="React and TypeScript UI work."),
    raw("nurse", "Pflegefachkraft (m/w/d)", tags=["Pflege"], location="Hamburg", description="Pflege im Krankenhaus, Schichtdienst."),
    raw("data-eng", "Data Engineer", tags=["Python", "Airflow", "Spark"], remote=True, description="Build batch pipelines with Airflow."),
    raw("sales", "Account Executive SaaS", tags=["Sales"], location="Munich", description="Sell B2B software to mid-size companies."),
]


@pytest.fixture
def conn():
    admin = db.connect()
    if not admin.execute("select 1 from pg_database where datname = %s", (TEST_DB,)).fetchone():
        admin.execute(f"create database {TEST_DB}")
    info = admin.info
    c = db.connect(f"postgresql://{info.user}:{info.password}@{info.host}:{info.port}/{TEST_DB}")
    db.migrate(c)
    c.execute("truncate jobs, users, memories, interactions, ingest_runs restart identity cascade")
    yield c
    c.close()


@pytest.fixture
def seeded(conn):
    ingest.sync(conn, FEED, complete=True)
    return conn


def job_id(conn, slug: str) -> int:
    return conn.execute("select id from jobs where source_id = %s", (slug,)).fetchone()["id"]
