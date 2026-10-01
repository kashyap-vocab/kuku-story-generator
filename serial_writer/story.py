"""Creating a story and reading or changing its top-level settings."""

from __future__ import annotations

import sqlite3

from .db import transaction

REVIEW_MODES = ("every_episode", "every_n", "on_issues", "arc_end")


def create_story(
    conn: sqlite3.Connection,
    premise: str,
    *,
    total_episodes: int = 200,
    review_mode: str = "every_episode",
    every_n: int | None = None,
) -> int:
    """A new story starts in planning, with the reviewer's chosen review setting."""
    premise = premise.strip()
    if not premise:
        raise ValueError("premise is empty")
    _check_review(review_mode, every_n)
    with transaction(conn):
        story_id = conn.execute(
            "INSERT INTO stories (premise, total_episodes) VALUES (?, ?)", (premise, total_episodes)
        ).lastrowid
        conn.execute(
            "INSERT INTO review_settings (story_id, mode, every_n) VALUES (?, ?, ?)",
            (story_id, review_mode, every_n),
        )
    return story_id


def set_review_setting(
    conn: sqlite3.Connection, story_id: int, mode: str, every_n: int | None = None,
    effective_from_ep: int = 1,
) -> None:
    """Adds a new setting; the old ones stay as history."""
    _check_review(mode, every_n)
    conn.execute(
        "INSERT INTO review_settings (story_id, mode, every_n, effective_from_ep) VALUES (?, ?, ?, ?)",
        (story_id, mode, every_n, effective_from_ep),
    )


def current_review_setting(conn: sqlite3.Connection, story_id: int) -> sqlite3.Row:
    return conn.execute(
        "SELECT * FROM review_settings WHERE story_id = ? ORDER BY id DESC LIMIT 1", (story_id,)
    ).fetchone()


def get_story(conn: sqlite3.Connection, story_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
    if row is None:
        raise KeyError(f"story {story_id} not found")
    return row


def set_status(conn: sqlite3.Connection, story_id: int, status: str) -> None:
    conn.execute(
        "UPDATE stories SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (status, story_id),
    )


def _check_review(mode: str, every_n: int | None) -> None:
    if mode not in REVIEW_MODES:
        raise ValueError(f"review mode must be one of {REVIEW_MODES}")
    if mode == "every_n" and (every_n is None or every_n < 1):
        raise ValueError("every_n mode needs a positive every_n")
