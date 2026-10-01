"""Reading and writing story rules and plan versions.

A plan is never edited in place. Every edit or redo copies the current version
into a new one (leaving out whatever is being rebuilt), so the history of what
the plan was, and why it changed, is always kept.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import to_json

# ---------------------------------------------------------------- story rules


def save_bible(
    conn: sqlite3.Connection, story_id: int, bible: dict[str, Any], *, created_by: str,
    reason: str | None = None,
) -> int:
    """Store a new version of the story rules and register its cast. Call inside a transaction."""
    version = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM bible_versions WHERE story_id = ?", (story_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE bible_versions SET status = 'superseded' WHERE story_id = ? AND status = 'draft'",
        (story_id,),
    )
    bible_id = conn.execute(
        "INSERT INTO bible_versions (story_id, version, content, reason, created_by) VALUES (?, ?, ?, ?, ?)",
        (story_id, version, to_json(bible), reason, created_by),
    ).lastrowid
    for c in bible["cast"]:
        upsert_character(conn, story_id, c["name"], c["role"], c["description"], c["importance"])
    return bible_id


def load_bible(conn: sqlite3.Connection, bible_version_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT content FROM bible_versions WHERE id = ?", (bible_version_id,)).fetchone()
    return json.loads(row["content"])


def latest_bible_id(conn: sqlite3.Connection, story_id: int) -> int | None:
    row = conn.execute(
        "SELECT id FROM bible_versions WHERE story_id = ? AND status IN ('draft', 'approved') "
        "ORDER BY version DESC LIMIT 1",
        (story_id,),
    ).fetchone()
    return row["id"] if row else None


def upsert_character(
    conn: sqlite3.Connection, story_id: int, name: str, role: str, description: str, importance: str
) -> int:
    conn.execute(
        """INSERT INTO characters (story_id, name, role, description, importance) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (story_id, name) DO UPDATE SET
               role = excluded.role, description = excluded.description, importance = excluded.importance""",
        (story_id, name, role, description, importance),
    )
    return conn.execute(
        "SELECT id FROM characters WHERE story_id = ? AND name = ?", (story_id, name)
    ).fetchone()["id"]


# ---------------------------------------------------------------- plan versions


def new_plan_version(
    conn: sqlite3.Connection, story_id: int, bible_version_id: int, *, created_by: str,
    reason: str | None = None, parent_id: int | None = None,
) -> int:
    """A new, empty draft plan. Older drafts are marked superseded. Call inside a transaction."""
    version = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM plan_versions WHERE story_id = ?", (story_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE plan_versions SET status = 'superseded' WHERE story_id = ? AND status = 'draft'",
        (story_id,),
    )
    return conn.execute(
        """INSERT INTO plan_versions (story_id, version, parent_id, bible_version_id, reason, created_by)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (story_id, version, parent_id, bible_version_id, reason, created_by),
    ).lastrowid


def latest_draft_plan_id(conn: sqlite3.Connection, story_id: int) -> int | None:
    row = conn.execute(
        "SELECT id FROM plan_versions WHERE story_id = ? AND status = 'draft' ORDER BY version DESC LIMIT 1",
        (story_id,),
    ).fetchone()
    return row["id"] if row else None


def get_plan_version(conn: sqlite3.Connection, plan_version_id: int) -> sqlite3.Row:
    return conn.execute("SELECT * FROM plan_versions WHERE id = ?", (plan_version_id,)).fetchone()


def seed_from_bible(conn: sqlite3.Connection, plan_version_id: int, story_id: int, bible: dict[str, Any]) -> None:
    """Give a new plan the bible's threads and cast. Call inside a transaction."""
    for t in bible["threads"]:
        conn.execute(
            "INSERT INTO plan_threads (plan_version_id, key, title, question, source) VALUES (?, ?, ?, ?, 'bible')",
            (plan_version_id, t["key"], t["title"], t["question"]),
        )
    for c in bible["cast"]:
        char_id = conn.execute(
            "SELECT id FROM characters WHERE story_id = ? AND name = ?", (story_id, c["name"])
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO plan_character_arcs (plan_version_id, character_id, arc, source) VALUES (?, ?, ?, 'bible')",
            (plan_version_id, char_id, c["arc"]),
        )


def copy_plan(
    conn: sqlite3.Connection, src: int, dst: int, *, keep_acts: bool = True,
    skip_arcs_of_acts: set[int] = frozenset(), skip_beats_of_arcs: set[int] = frozenset(),
) -> None:
    """Copy a plan into another version, leaving out the parts about to be rebuilt.
    Threads and characters added by a left-out arc are left out with it."""
    kept_arcs: set[int] = set()
    if keep_acts:
        conn.execute(
            """INSERT INTO plan_acts (plan_version_id, act_no, title, goal, turning_point, start_ep, end_ep, details)
               SELECT ?, act_no, title, goal, turning_point, start_ep, end_ep, details FROM plan_acts
               WHERE plan_version_id = ?""",
            (dst, src),
        )
        for arc in conn.execute("SELECT * FROM plan_arcs WHERE plan_version_id = ?", (src,)).fetchall():
            if arc["act_no"] in skip_arcs_of_acts:
                continue
            kept_arcs.add(arc["arc_no"])
            conn.execute(
                """INSERT INTO plan_arcs (plan_version_id, arc_no, act_no, title, goal, turning_point,
                       start_ep, end_ep, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (dst, arc["arc_no"], arc["act_no"], arc["title"], arc["goal"], arc["turning_point"],
                 arc["start_ep"], arc["end_ep"], arc["details"]),
            )
            if arc["arc_no"] in skip_beats_of_arcs:
                continue
            conn.execute(
                """INSERT INTO plan_beats (plan_version_id, ep_no, beat, hook, characters, threads, is_turning_point)
                   SELECT ?, ep_no, beat, hook, characters, threads, is_turning_point FROM plan_beats
                   WHERE plan_version_id = ? AND ep_no BETWEEN ? AND ?""",
                (dst, src, arc["start_ep"], arc["end_ep"]),
            )

    def kept(source: str) -> bool:
        return source == "bible" or (source.startswith("arc:") and int(source[4:]) in kept_arcs)

    for t in conn.execute("SELECT * FROM plan_threads WHERE plan_version_id = ?", (src,)).fetchall():
        if kept(t["source"]):
            conn.execute(
                "INSERT INTO plan_threads (plan_version_id, key, title, question, source) VALUES (?, ?, ?, ?, ?)",
                (dst, t["key"], t["title"], t["question"], t["source"]),
            )
    for c in conn.execute("SELECT * FROM plan_character_arcs WHERE plan_version_id = ?", (src,)).fetchall():
        if kept(c["source"]):
            conn.execute(
                "INSERT INTO plan_character_arcs (plan_version_id, character_id, arc, source) VALUES (?, ?, ?, ?)",
                (dst, c["character_id"], c["arc"], c["source"]),
            )


# ---------------------------------------------------------------- loading


def load_acts(conn: sqlite3.Connection, pv: int) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM plan_acts WHERE plan_version_id = ? ORDER BY act_no", (pv,)).fetchall()
    return [{**dict(r), "details": json.loads(r["details"])} for r in rows]


def load_arcs(conn: sqlite3.Connection, pv: int) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM plan_arcs WHERE plan_version_id = ? ORDER BY arc_no", (pv,)).fetchall()
    return [{**dict(r), "details": json.loads(r["details"])} for r in rows]


def load_beats(
    conn: sqlite3.Connection, pv: int, start: int | None = None, end: int | None = None
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM plan_beats WHERE plan_version_id = ? AND ep_no BETWEEN ? AND ? ORDER BY ep_no",
        (pv, start if start is not None else 0, end if end is not None else 10**9),
    ).fetchall()
    return [
        {**dict(r), "characters": json.loads(r["characters"]), "threads": json.loads(r["threads"])}
        for r in rows
    ]


def load_threads(conn: sqlite3.Connection, pv: int) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM plan_threads WHERE plan_version_id = ? ORDER BY id", (pv,)).fetchall()
    return [dict(r) for r in rows]


def load_cast(conn: sqlite3.Connection, pv: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT c.id, c.name, c.role, c.description, c.importance, pca.arc, pca.source
           FROM plan_character_arcs pca JOIN characters c ON c.id = pca.character_id
           WHERE pca.plan_version_id = ? ORDER BY pca.id""",
        (pv,),
    ).fetchall()
    return [dict(r) for r in rows]


def available(source: str, up_to_arc: int) -> bool:
    """Is a thread or character with this source in play by arc `up_to_arc`?"""
    return source == "bible" or int(source.split(":")[1]) <= up_to_arc


def thread_lifecycle(
    moves: list[tuple[int, str]], state: str = "not_opened"
) -> tuple[list[tuple[int, str, str]], str]:
    """Walk one thread's moves (ep, event) in order from `state`.
    Returns (problems as (ep, severity, message), final state)."""
    problems: list[tuple[int, str, str]] = []
    for ep, event in moves:
        if event == "open":
            if state == "open":
                problems.append((ep, "minor", "is opened again while already open"))
            elif state == "resolved":
                problems.append((ep, "must_fix", "is opened again after it was resolved"))
            state = "open"
        elif state == "not_opened":
            problems.append((ep, "must_fix", f"has '{event}' before it was opened"))
            state = "resolved" if event == "resolve" else "open"
        elif state == "resolved":
            problems.append((ep, "must_fix", f"has '{event}' after it was resolved"))
        elif event == "resolve":
            state = "resolved"
    return problems, state


def thread_status(
    threads: list[dict[str, Any]], beats: list[dict[str, Any]], before_ep: int
) -> list[dict[str, Any]]:
    """Where each thread stands just before `before_ep`, from the plan lines."""
    out = []
    for t in threads:
        opened = last = None
        state = "not_opened"
        for b in beats:
            if b["ep_no"] >= before_ep:
                break
            for m in b["threads"]:
                if m["key"] != t["key"]:
                    continue
                if m["event"] == "open" and state != "open":
                    state, opened = "open", b["ep_no"]
                elif m["event"] == "resolve":
                    state = "resolved"
                last = b["ep_no"]
        out.append({**t, "state": state, "opened": opened, "last": last})
    return out
