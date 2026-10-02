import sqlite3

import pytest

from serial_writer.db import init_db


def _story(conn) -> int:
    return conn.execute("INSERT INTO stories (premise) VALUES ('p')").lastrowid


def _episode(conn, story_id, ep_no, version, status="draft") -> int:
    return conn.execute(
        "INSERT INTO episode_versions (story_id, ep_no, version, text, word_count, created_by, status) "
        "VALUES (?, ?, ?, 'text', 1, 'model', ?)",
        (story_id, ep_no, version, status),
    ).lastrowid


def test_init_is_repeatable(conn):
    init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_only_one_approved_version_per_episode(conn):
    s = _story(conn)
    _episode(conn, s, 1, 1, "approved")
    _episode(conn, s, 1, 2, "draft")
    with pytest.raises(sqlite3.IntegrityError):
        _episode(conn, s, 1, 3, "approved")


def test_episode_text_cannot_be_overwritten_or_deleted(conn):
    s = _story(conn)
    ev = _episode(conn, s, 1, 1)
    with pytest.raises(sqlite3.IntegrityError, match="frozen"):
        conn.execute("UPDATE episode_versions SET text = 'new' WHERE id = ?", (ev,))
    with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        conn.execute("DELETE FROM episode_versions WHERE id = ?", (ev,))
    # Moving the status on is allowed.
    conn.execute("UPDATE episode_versions SET status = 'approved' WHERE id = ?", (ev,))


def test_human_records_are_append_only(conn):
    s = _story(conn)
    fb = conn.execute(
        "INSERT INTO feedback (story_id, ep_no, action, text) VALUES (?, 3, 'note', 'slow down the romance')",
        (s,),
    ).lastrowid
    d = conn.execute(
        "INSERT INTO directives (story_id, text, kind, from_feedback_id, from_ep) "
        "VALUES (?, 'slow down the romance', 'pacing', ?, 3)",
        (s, fb),
    ).lastrowid
    for sql in ("DELETE FROM feedback", "UPDATE feedback SET text = 'x'",
                "DELETE FROM directives", "UPDATE directives SET text = 'x'"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(sql)

    conn.execute("INSERT INTO directive_status (directive_id, status) VALUES (?, 'active')", (d,))
    conn.execute("INSERT INTO directive_status (directive_id, status, reason) VALUES (?, 'paused', 'arc 3')", (d,))
    row = conn.execute("SELECT status, status_reason FROM current_directives WHERE id = ?", (d,)).fetchone()
    assert (row["status"], row["status_reason"]) == ("paused", "arc 3")
    # Full history is still there.
    assert conn.execute("SELECT COUNT(*) FROM directive_status").fetchone()[0] == 2


def test_memory_only_counts_from_approved_episodes(conn):
    s = _story(conn)
    approved = _episode(conn, s, 1, 1, "approved")
    draft = _episode(conn, s, 2, 1, "draft")
    for src, text in ((approved, "the shop is on Elm St"), (draft, "draft-only fact"), (None, "rule from the bible")):
        conn.execute(
            "INSERT INTO facts (story_id, source_version_id, category, text) VALUES (?, ?, 'world', ?)",
            (s, src, text),
        )
    live = {r["text"] for r in conn.execute("SELECT text FROM live_facts")}
    assert live == {"the shop is on Elm St", "rule from the bible"}

    # Rejecting the approved episode removes its memory without deleting anything.
    conn.execute("UPDATE episode_versions SET status = 'rejected' WHERE id = ?", (approved,))
    live = {r["text"] for r in conn.execute("SELECT text FROM live_facts")}
    assert live == {"rule from the bible"}
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3


def test_replaced_facts_drop_out_but_are_kept(conn):
    s = _story(conn)
    old = conn.execute(
        "INSERT INTO facts (story_id, category, text) VALUES (?, 'world', 'Maya lives alone')", (s,)
    ).lastrowid
    new = conn.execute(
        "INSERT INTO facts (story_id, category, text) VALUES (?, 'world', 'Maya moved in with Ravi')", (s,)
    ).lastrowid
    conn.execute("UPDATE facts SET replaced_by = ? WHERE id = ?", (new, old))
    assert [r["text"] for r in conn.execute("SELECT text FROM live_facts")] == ["Maya moved in with Ravi"]


def test_relationship_pairs_stored_one_way(conn):
    s = _story(conn)
    a = conn.execute("INSERT INTO characters (story_id, name) VALUES (?, 'A')", (s,)).lastrowid
    b = conn.execute("INSERT INTO characters (story_id, name) VALUES (?, 'B')", (s,)).lastrowid
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO relationships (story_id, char_a, char_b, ep_no, state) VALUES (?, ?, ?, 1, 'x')",
            (s, max(a, b), min(a, b)),
        )
