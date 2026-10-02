"""SQLite connection and setup."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 2

# Changes to tables that already exist in older databases. New tables need no
# entry: the schema script creates whatever is missing.
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE episode_versions ADD COLUMN context_id INTEGER REFERENCES episode_contexts(id)",
        "ALTER TABLE episode_versions ADD COLUMN summary TEXT",
        "ALTER TABLE episode_versions ADD COLUMN story_time TEXT",
        "ALTER TABLE episode_versions ADD COLUMN outline TEXT",
        "ALTER TABLE threads ADD COLUMN key TEXT",
    ],
}


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the story database with the settings every connection needs."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Autocommit mode; multi-statement writes go through `transaction()`.
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets the web app read while the writer is generating.
    conn.execute("PRAGMA journal_mode = WAL")
    # Wait rather than fail if another connection is briefly writing.
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables, views and guards. Safe to run on an existing database."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema v{current} is newer than this code (v{SCHEMA_VERSION})."
        )
    existing = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'stories'").fetchone() is not None
    if existing:
        for version in range(current + 1, SCHEMA_VERSION + 1):
            for stmt in MIGRATIONS.get(version, []):
                conn.execute(stmt)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """All-or-nothing block of writes."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
