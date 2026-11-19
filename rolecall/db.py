import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

load_dotenv()

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
DEFAULT_URL = "postgresql://rolecall:rolecall@localhost:5544/rolecall"


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or os.getenv("DATABASE_URL", DEFAULT_URL), row_factory=dict_row, autocommit=True)
    # The vector type only exists after the extension does, so create it before registering.
    conn.execute("create extension if not exists vector")
    register_vector(conn)
    return conn


def migrate(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text())


def ensure_user(conn: psycopg.Connection, user_id: str) -> None:
    conn.execute("insert into users (id) values (%s) on conflict do nothing", (user_id,))
