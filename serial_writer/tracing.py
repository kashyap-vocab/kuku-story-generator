"""Run, step and model-call logging: what happened, what it cost, how long it took."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from .db import to_json


def start_run(conn: sqlite3.Connection, kind: str, story_id: int | None = None) -> int:
    cur = conn.execute("INSERT INTO runs (story_id, kind) VALUES (?, ?)", (story_id, kind))
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, note: str | None = None) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, note = ?, ended_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
        "WHERE id = ?",
        (status, note, run_id),
    )


def log_step(
    conn: sqlite3.Connection,
    node: str,
    decision: str,
    *,
    run_id: int | None = None,
    story_id: int | None = None,
    ep_no: int | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO steps (run_id, story_id, ep_no, node, decision, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, story_id, ep_no, node, decision, to_json(detail) if detail is not None else None),
    )


@dataclass
class CallRecord:
    node: str
    attempt: int
    model: str
    temperature: float | None
    max_tokens: int | None
    messages: list[dict[str, str]]
    status: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None
    error: str | None = None
    response: str | None = None
    run_id: int | None = None
    story_id: int | None = None
    ep_no: int | None = None


def log_call(conn: sqlite3.Connection, rec: CallRecord) -> None:
    conn.execute(
        """INSERT INTO llm_calls (run_id, story_id, ep_no, node, attempt, model, temperature,
               max_tokens, prompt_tokens, completion_tokens, latency_ms, status, finish_reason,
               error, messages, response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            rec.run_id, rec.story_id, rec.ep_no, rec.node, rec.attempt, rec.model,
            rec.temperature, rec.max_tokens, rec.prompt_tokens, rec.completion_tokens,
            rec.latency_ms, rec.status, rec.finish_reason, rec.error,
            to_json(rec.messages), rec.response,
        ),
    )


def episode_tokens_used(conn: sqlite3.Connection, story_id: int, ep_no: int, since: str | None = None) -> int:
    """Tokens spent on one episode (since `since`, if given), failed attempts included."""
    row = conn.execute(
        "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM llm_calls "
        "WHERE story_id = ? AND ep_no = ? AND created_at >= ?",
        (story_id, ep_no, since or ""),
    ).fetchone()
    return row[0]


def usage_by_node(conn: sqlite3.Connection, story_id: int) -> list[sqlite3.Row]:
    """Calls, tokens and time per pipeline step, for the cost report."""
    return conn.execute(
        """SELECT node,
                  COUNT(*)                         AS calls,
                  SUM(status <> 'ok')              AS failed,
                  SUM(prompt_tokens)               AS prompt_tokens,
                  SUM(completion_tokens)           AS completion_tokens,
                  SUM(latency_ms)                  AS latency_ms
           FROM llm_calls WHERE story_id = ?
           GROUP BY node ORDER BY node""",
        (story_id,),
    ).fetchall()
