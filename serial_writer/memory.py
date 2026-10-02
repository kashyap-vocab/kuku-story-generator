"""The story's memory: reading it, packing it for one episode, and saving what
an episode adds to it.

Everything reads through the live_* views, so only approved episodes count.
What a draft adds is saved against the draft's version right away, which lets
the reviewer see exactly what will be remembered; it becomes real memory only
when that version is approved.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import to_json
from .plan_prompts import render_bible
from .plan_store import load_acts, load_arcs, load_beats, load_bible, load_threads
from .similarity import contains_quote, words

RECENT_ONE_LINERS = 5
STORY_SO_FAR_MAX = 40
THREAD_OVERDUE_EPS = 15
CHARACTER_ABSENT_EPS = 20
LOOKUP_FACTS = 15
RECENT_FACT_EPS = 3
LOOKUP_KEY_LINES = 8
KNOWS_SHOWN = 6


def approved_episodes(conn: sqlite3.Connection, story_id: int, before_ep: int | None = None) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM episode_versions WHERE story_id = ? AND status = 'approved' AND ep_no < ? ORDER BY ep_no",
        (story_id, before_ep if before_ep is not None else 10**9),
    ).fetchall()
    return [dict(r) for r in rows]


def next_episode_no(conn: sqlite3.Connection, story_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(ep_no), 0) FROM episode_versions WHERE story_id = ? AND status = 'approved'", (story_id,)
    ).fetchone()
    return row[0] + 1


def character_states(conn: sqlite3.Connection, story_id: int, before_ep: int) -> dict[str, dict[str, Any]]:
    """Latest live state of each character before `before_ep`, by name."""
    rows = conn.execute(
        """SELECT c.name, c.importance, cs.* FROM live_character_states cs
           JOIN characters c ON c.id = cs.character_id
           WHERE c.story_id = ? AND cs.ep_no < ? ORDER BY cs.ep_no, cs.id""",
        (story_id, before_ep),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["name"]] = {**dict(r), "knows": json.loads(r["knows"])}
    return out


def relationships(conn: sqlite3.Connection, story_id: int, before_ep: int) -> list[dict[str, Any]]:
    """Latest live state of each pair."""
    rows = conn.execute(
        """SELECT r.*, a.name AS a_name, b.name AS b_name FROM live_relationships r
           JOIN characters a ON a.id = r.char_a JOIN characters b ON b.id = r.char_b
           WHERE r.story_id = ? AND r.ep_no < ? ORDER BY r.ep_no, r.id""",
        (story_id, before_ep),
    ).fetchall()
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for r in rows:
        latest[(r["char_a"], r["char_b"])] = dict(r)
    return list(latest.values())


def thread_states(conn: sqlite3.Connection, story_id: int, before_ep: int) -> list[dict[str, Any]]:
    """Each live thread with its events so far and where it stands."""
    threads = conn.execute(
        "SELECT * FROM live_threads WHERE story_id = ? ORDER BY id", (story_id,)
    ).fetchall()
    out = []
    for t in threads:
        events = [dict(e) for e in conn.execute(
            "SELECT * FROM live_thread_events WHERE thread_id = ? AND ep_no < ? ORDER BY ep_no, id",
            (t["id"], before_ep),
        )]
        if not events:
            continue
        state = "resolved" if events[-1]["kind"] == "resolved" else "open"
        out.append({**dict(t), "events": events, "state": state, "last_ep": events[-1]["ep_no"]})
    return out


def directives(conn: sqlite3.Connection, story_id: int, status: str | None = "active") -> list[dict[str, Any]]:
    sql = "SELECT * FROM current_directives WHERE story_id = ?"
    args: list[Any] = [story_id]
    if status:
        sql += " AND status = ?"
        args.append(status)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", args)]


def summaries(conn: sqlite3.Connection, story_id: int, before_ep: int) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM summaries WHERE story_id = ? AND level = 'arc' AND end_ep < ? AND stale = 0 ORDER BY start_ep",
        (story_id, before_ep),
    )]


def proposed_memory(conn: sqlite3.Connection, version_id: int) -> dict[str, list[dict[str, Any]]]:
    """What approving this version would add to memory, for the reviewer."""
    q = lambda sql: [dict(r) for r in conn.execute(sql, (version_id,))]
    return {
        "states": q("""SELECT c.name, cs.* FROM character_states cs JOIN characters c ON c.id = cs.character_id
                       WHERE cs.source_version_id = ? ORDER BY cs.id"""),
        "relationships": q("""SELECT a.name AS a_name, b.name AS b_name, r.* FROM relationships r
                              JOIN characters a ON a.id = r.char_a JOIN characters b ON b.id = r.char_b
                              WHERE r.source_version_id = ? ORDER BY r.id"""),
        "facts": q("SELECT * FROM facts WHERE source_version_id = ? ORDER BY id"),
        "thread_events": q("""SELECT t.key, t.title, te.* FROM thread_events te JOIN threads t ON t.id = te.thread_id
                              WHERE te.source_version_id = ? ORDER BY te.id"""),
        "key_lines": q("""SELECT c.name AS speaker, k.* FROM key_lines k LEFT JOIN characters c ON c.id = k.speaker_id
                          WHERE k.source_version_id = ? ORDER BY k.id"""),
        "new_characters": q("SELECT * FROM characters WHERE source_version_id = ? ORDER BY id"),
    }


def build_pack(conn: sqlite3.Connection, story_id: int, ep: int, pv: int) -> tuple[str, dict[str, list[int]]]:
    """The fixed-size slice of memory the model gets for episode `ep`, picked by code.

    Ordered from what changes least to what changes most, so the model server
    can reuse the start of the prompt between calls. Returns (text, refs): refs
    are the ids of every memory row in the pack."""
    plan = conn.execute("SELECT bible_version_id FROM plan_versions WHERE id = ?", (pv,)).fetchone()
    bible = load_bible(conn, plan["bible_version_id"])
    refs: dict[str, list[int]] = {"episodes": [], "states": [], "relationships": [], "facts": [],
                                  "thread_events": [], "key_lines": [], "summaries": [], "directives": []}
    parts: list[str] = []

    parts.append("=== STORY RULES ===\n" + render_bible(bible))
    active = directives(conn, story_id)
    refs["directives"] = [d["id"] for d in active]
    parts.append("=== HUMAN INSTRUCTIONS (always follow these; they override the plan) ===\n" + (
        "\n".join(f"- {d['text']}" for d in active) if active else "(none)"))

    act = next(a for a in load_acts(conn, pv) if a["start_ep"] <= ep <= a["end_ep"])
    arc = next(a for a in load_arcs(conn, pv) if a["start_ep"] <= ep <= a["end_ep"])
    beats = {b["ep_no"]: b for b in load_beats(conn, pv, ep - 2, ep + 3)}
    plan_lines = []
    for n in range(ep - 2, ep + 4):
        if n not in beats:
            continue
        b = beats[n]
        tag = ">>> THIS EPISODE" if n == ep else ("(already written)" if n < ep else "(coming later; do not get there yet)")
        plan_lines.append(f"Ep {n} {tag}: {b['beat']} | Hook: {b['hook']}")
    parts.append(
        f"=== THE PLAN ===\nACT {act['act_no']} (ep {act['start_ep']}-{act['end_ep']}): {act['title']}: {act['goal']}\n"
        f"ARC {arc['arc_no']} (ep {arc['start_ep']}-{arc['end_ep']}): {arc['title']}: {arc['goal']}\n"
        + "\n".join(plan_lines)
    )

    done = approved_episodes(conn, story_id, ep)
    arc_sums = summaries(conn, story_id, ep)
    refs["summaries"] = [s["id"] for s in arc_sums]
    covered_to = max((s["end_ep"] for s in arc_sums), default=0)
    recent_from = ep - 1 - RECENT_ONE_LINERS
    older = [e for e in done if covered_to < e["ep_no"] < recent_from][-STORY_SO_FAR_MAX:]
    so_far = [f"Eps {s['start_ep']}-{s['end_ep']}: {s['narrative']}" for s in arc_sums]
    so_far += [f"Ep {e['ep_no']}: {e['one_line']}" for e in older]
    parts.append("=== THE STORY SO FAR ===\n" + ("\n".join(so_far) if so_far else "(nothing before the recent episodes)"))

    recent = [e for e in done if e["ep_no"] >= recent_from]
    refs["episodes"] = [e["id"] for e in recent]
    if recent:
        lines = [f"Ep {e['ep_no']} ({e['story_time'] or 'time not noted'}): {e['one_line']}" for e in recent[:-1]]
        last = recent[-1]
        parts.append("=== RECENT EPISODES ===\n" + "\n".join(lines) +
                     f"\n\nLAST EPISODE IN FULL (ep {last['ep_no']}, ends at {last['story_time'] or 'time not noted'}):\n{last['text']}")
    else:
        parts.append("=== RECENT EPISODES ===\n(this is the first episode)")

    beat = beats.get(ep) or {"characters": [], "threads": []}
    states = character_states(conn, story_id, ep)
    names = list(beat["characters"])
    who = []
    for name in names:
        s = states.get(name)
        if s is None:
            who.append(f"- {name}: not in the story yet (first appearance).")
            continue
        refs["states"].append(s["id"])
        if s["status"] == "dead":
            who.append(f"- {name}: DEAD since ep {s['ep_no']}. The plan was written before that: write around it, "
                       f"don't bring them back (unless the world rules allow it).")
            continue
        knows = "; ".join(s["knows"][-KNOWS_SHOWN:]) or "nothing special"
        who.append(f"- {name}: {s['status'].upper()}, last at {s['location'] or 'unknown'} (ep {s['ep_no']}). "
                   f"Wants: {s['goal'] or '-'}. Knows: {knows}.")
    dead = [n for n, s in states.items() if s["status"] == "dead"]
    rels = [r for r in relationships(conn, story_id, ep) if r["a_name"] in names or r["b_name"] in names]
    refs["relationships"] = [r["id"] for r in rels]
    threads = thread_states(conn, story_id, ep)
    in_beat = {m["key"] for m in beat["threads"]}
    plan_threads = {t["key"]: t for t in load_threads(conn, pv)}
    thread_lines = []
    for t in threads:
        if t["key"] in in_beat:
            refs["thread_events"] += [e["id"] for e in t["events"]]
            history = "; ".join(f"ep {e['ep_no']} {e['kind']}: {e['note'] or ''}".strip() for e in t["events"][-5:])
            thread_lines.append(f"- [{t['key']}] {t['title']} ({t['state'].upper()}): {history}")
    for key in in_beat - {t["key"] for t in threads}:
        if key in plan_threads:
            thread_lines.append(f"- [{key}] {plan_threads[key]['title']}: not raised yet. Question: {plan_threads[key]['question']}")
    overdue = [t for t in threads if t["state"] == "open" and t["key"] not in in_beat and ep - t["last_ep"] > THREAD_OVERDUE_EPS]
    absent = [n for n, s in states.items() if s["importance"] == "major" and s["status"] != "dead"
              and n not in names and ep - s["ep_no"] > CHARACTER_ABSENT_EPS]
    section = ["=== IN THIS EPISODE ===", "PEOPLE:", *(who or ["(none named)"])]
    if rels:
        section += ["RELATIONSHIPS:", *[f"- {r['a_name']} & {r['b_name']}: {r['state']} (ep {r['ep_no']})" for r in rels]]
    section += ["THREADS:", *(thread_lines or ["(none)"])]
    if dead:
        section.append("DEAD (can't act or speak unless the world rules allow it): " + ", ".join(dead))
    if overdue:
        section.append("THREADS LEFT QUIET TOO LONG (mention one if it fits): " +
                       "; ".join(f"[{t['key']}] since ep {t['last_ep']}" for t in overdue))
    if absent:
        section.append("MAIN CHARACTERS NOT SEEN FOR A WHILE: " + ", ".join(absent))
    parts.append("\n".join(section))

    terms = names + [n.split()[0] for n in names if " " in n] + \
        [plan_threads[k]["title"] for k in in_beat if k in plan_threads]
    facts = _lookup_facts(conn, story_id, ep, terms, f"{beat.get('beat', '')} {beat.get('hook', '')}")
    refs["facts"] = [f["id"] for f in facts]
    lines = _lookup_key_lines(conn, story_id, ep, names)
    refs["key_lines"] = [k["id"] for k in lines]
    looked = ["=== FACTS ALREADY ESTABLISHED (never contradict these) ==="]
    looked += [f"- (ep {f['ep_no']}{', ' + f['story_time'] if f['story_time'] else ''}) {f['text']}" for f in facts] or ["(none yet)"]
    if lines:
        looked += ["KEY LINES SAID BEFORE:", *[f"- ep {k['ep_no']}, {k['speaker'] or 'narration'}: \"{k['line']}\"" for k in lines]]
    parts.append("\n".join(looked))

    return "\n\n".join(parts), refs


def _lookup_facts(
    conn: sqlite3.Connection, story_id: int, ep: int, terms: list[str], plan_text: str
) -> list[dict[str, Any]]:
    """Facts worth knowing for this episode: everything from the last few episodes,
    older facts naming these people and threads or sharing words with the plan
    line (so "the ledger" finds facts about the ledger), and the latest timeline."""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM live_facts WHERE story_id = ? AND (ep_no IS NULL OR ep_no < ?) ORDER BY ep_no DESC, id DESC",
        (story_id, ep),
    )]
    names = [t.casefold() for t in terms if t]
    plan_words = words(plan_text)
    recent = [f for f in rows if (f["ep_no"] or 0) >= ep - RECENT_FACT_EPS]
    about = [f for f in rows if f not in recent and (
        any(n in f["text"].casefold() for n in names) or len(words(f["text"]) & plan_words) >= 2)]
    picked = (recent + about)[:LOOKUP_FACTS]
    timeline = [f for f in rows if f["category"] == "timeline" and f not in picked][:3]
    return sorted(picked + timeline, key=lambda f: (f["ep_no"] or 0, f["id"]))


def _lookup_key_lines(conn: sqlite3.Connection, story_id: int, ep: int, names: list[str]) -> list[dict[str, Any]]:
    rows = [dict(r) for r in conn.execute(
        """SELECT c.name AS speaker, k.* FROM live_key_lines k LEFT JOIN characters c ON c.id = k.speaker_id
           WHERE k.story_id = ? AND k.ep_no < ? ORDER BY k.ep_no DESC, k.id DESC""",
        (story_id, ep),
    )]
    picked = [k for k in rows if k["speaker"] in names or any(n.split()[0] in k["line"] for n in names)]
    return sorted(picked[:LOOKUP_KEY_LINES], key=lambda k: (k["ep_no"], k["id"]))


def save_context(conn: sqlite3.Connection, story_id: int, ep: int, pv: int, pack: str, refs: dict[str, Any]) -> int:
    return conn.execute(
        "INSERT INTO episode_contexts (story_id, ep_no, plan_version_id, pack, refs) VALUES (?, ?, ?, ?, ?)",
        (story_id, ep, pv, pack, to_json(refs)),
    ).lastrowid


def save_memory(
    conn: sqlite3.Connection, story_id: int, ep: int, version_id: int, text: str, ext: dict[str, Any],
    plan_threads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Store an episode's memory updates against its version. Call inside a transaction.

    Every update must quote the episode; updates whose quote isn't in the text
    are dropped (and returned, so they can be logged). Returns what was kept and dropped."""
    ids = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM characters WHERE story_id = ?", (story_id,))}
    states = character_states(conn, story_id, ep)
    dropped: list[dict[str, Any]] = []
    kept = {"states": 0, "relationships": 0, "facts": 0, "thread_events": 0, "key_lines": 0, "new_characters": 0}

    def proven(kind: str, item: dict[str, Any], quote_field: str = "quote") -> bool:
        if contains_quote(text, item.get(quote_field) or ""):
            return True
        dropped.append({"kind": kind, **item})
        return False

    for c in ext.get("new_characters", []):
        name = c["name"].strip()
        if not name or not proven("new_character", c):
            continue
        if name in ids:
            conn.execute(
                """UPDATE characters SET source_version_id = ?, first_ep = ? WHERE id = ? AND source_version_id IS NOT NULL
                   AND source_version_id NOT IN (SELECT id FROM episode_versions WHERE status = 'approved')""",
                (version_id, ep, ids[name]),
            )
        else:
            ids[name] = conn.execute(
                """INSERT INTO characters (story_id, name, role, description, importance, first_ep, source_version_id)
                   VALUES (?, ?, ?, ?, 'minor', ?, ?)""",
                (story_id, name, c.get("role", ""), c.get("description", ""), ep, version_id),
            ).lastrowid
        kept["new_characters"] += 1
        conn.execute(
            """INSERT INTO character_states (character_id, ep_no, source_version_id, status, location, knows, goal)
               VALUES (?, ?, ?, 'alive', ?, '[]', NULL)""",
            (ids[name], ep, version_id, c.get("location") or None),
        )

    for u in ext.get("characters", []):
        if u["name"] not in ids or not proven("character", u):
            continue
        before = states.get(u["name"], {})
        knows = list(before.get("knows", []))
        knows += [k for k in u.get("learned", []) if k and k not in knows]
        conn.execute(
            """INSERT INTO character_states (character_id, ep_no, source_version_id, status, location, knows, goal, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ids[u["name"]], ep, version_id, u.get("status") or before.get("status", "alive"),
             u.get("location") or before.get("location"), to_json(knows),
             u.get("goal") or before.get("goal"), u.get("change") or None),
        )
        kept["states"] += 1

    for r in ext.get("relationships", []):
        if r["a"] not in ids or r["b"] not in ids or r["a"] == r["b"] or not proven("relationship", r):
            continue
        a, b = sorted((ids[r["a"]], ids[r["b"]]))
        conn.execute(
            "INSERT INTO relationships (story_id, char_a, char_b, ep_no, source_version_id, state) VALUES (?, ?, ?, ?, ?, ?)",
            (story_id, a, b, ep, version_id, r["state"]),
        )
        kept["relationships"] += 1

    for f in ext.get("facts", []):
        if not f.get("text") or not proven("fact", f):
            continue
        conn.execute(
            "INSERT INTO facts (story_id, ep_no, source_version_id, category, story_time, text) VALUES (?, ?, ?, ?, ?, ?)",
            (story_id, ep, version_id, f.get("category", "other"), ext.get("story_time") or None, f["text"]),
        )
        kept["facts"] += 1

    for t in ext.get("threads", []):
        if t["key"] not in plan_threads or not proven("thread", t):
            continue
        row = conn.execute(
            "SELECT id FROM live_threads WHERE story_id = ? AND key = ? ORDER BY id LIMIT 1", (story_id, t["key"])
        ).fetchone() or conn.execute(
            "SELECT id FROM threads WHERE story_id = ? AND key = ? AND source_version_id = ?",
            (story_id, t["key"], version_id),
        ).fetchone()
        if row is None:
            pt = plan_threads[t["key"]]
            thread_id = conn.execute(
                "INSERT INTO threads (story_id, key, title, description, opened_ep, source_version_id) VALUES (?, ?, ?, ?, ?, ?)",
                (story_id, t["key"], pt["title"], pt["question"], ep, version_id),
            ).lastrowid
        else:
            thread_id = row["id"]
        conn.execute(
            "INSERT INTO thread_events (thread_id, ep_no, source_version_id, kind, note) VALUES (?, ?, ?, ?, ?)",
            (thread_id, ep, version_id, t["event"], t.get("note") or None),
        )
        kept["thread_events"] += 1

    for k in ext.get("key_lines", []):
        if not proven("key_line", k, "line"):
            continue
        conn.execute(
            "INSERT INTO key_lines (story_id, ep_no, source_version_id, speaker_id, line, why) VALUES (?, ?, ?, ?, ?, ?)",
            (story_id, ep, version_id, ids.get(k.get("speaker", "")), k["line"], k.get("why") or None),
        )
        kept["key_lines"] += 1

    conn.execute(
        "UPDATE episode_versions SET one_line = ?, summary = ?, story_time = ? WHERE id = ?",
        (ext.get("one_line"), ext.get("summary"), ext.get("story_time") or None, version_id),
    )
    return {"kept": kept, "dropped": dropped}


def approve_version(conn: sqlite3.Connection, story_id: int, version_id: int) -> None:
    """Make this version the episode, and its memory live. Call inside a transaction."""
    ep = conn.execute("SELECT ep_no FROM episode_versions WHERE id = ?", (version_id,)).fetchone()["ep_no"]
    conn.execute(
        """UPDATE episode_versions SET status = 'superseded', decided_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
           WHERE story_id = ? AND ep_no = ? AND id <> ? AND status IN ('approved', 'draft', 'in_review')""",
        (story_id, ep, version_id),
    )
    conn.execute(
        "UPDATE episode_versions SET status = 'approved', decided_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (version_id,),
    )
    conn.execute(
        "UPDATE summaries SET stale = 1 WHERE story_id = ? AND ? BETWEEN start_ep AND end_ep AND stale = 0",
        (story_id, ep),
    )


def arcs_to_summarize(conn: sqlite3.Connection, story_id: int, pv: int) -> list[dict[str, Any]]:
    """Finished arcs (every episode approved) without a current summary."""
    written_to = next_episode_no(conn, story_id) - 1
    have = {(r["start_ep"], r["end_ep"]) for r in conn.execute(
        "SELECT start_ep, end_ep FROM summaries WHERE story_id = ? AND level = 'arc' AND stale = 0", (story_id,))}
    return [a for a in load_arcs(conn, pv) if a["end_ep"] <= written_to and (a["start_ep"], a["end_ep"]) not in have]


def arc_snapshot(conn: sqlite3.Connection, story_id: int, start: int, end: int) -> dict[str, Any]:
    """The summary's fixed sections, filled from the tables by code, so they can't drift."""
    states = character_states(conn, story_id, end + 1)
    threads = thread_states(conn, story_id, end + 1)
    span = (story_id, start, end)
    return {
        "characters": [
            {"name": n, "status": s["status"], "location": s["location"], "goal": s["goal"], "last_seen_ep": s["ep_no"]}
            for n, s in states.items()
        ],
        "open_threads": [
            {"key": t["key"], "title": t["title"], "opened_ep": t["events"][0]["ep_no"], "last_ep": t["last_ep"],
             "quiet_for": end - t["last_ep"]}
            for t in threads if t["state"] == "open"
        ],
        "resolved_here": [t["title"] for t in threads if t["state"] == "resolved" and start <= t["last_ep"] <= end],
        "timeline": [
            {"ep": e["ep_no"], "story_time": e["story_time"], "what": e["one_line"]}
            for e in approved_episodes(conn, story_id, end + 1) if e["ep_no"] >= start
        ],
        "key_lines": [
            {"ep": k["ep_no"], "speaker": k["speaker"], "line": k["line"]}
            for k in conn.execute(
                """SELECT c.name AS speaker, k.* FROM live_key_lines k LEFT JOIN characters c ON c.id = k.speaker_id
                   WHERE k.story_id = ? AND k.ep_no BETWEEN ? AND ? ORDER BY k.ep_no""", span)
        ],
        "human_review_points": [
            {"ep": f["ep_no"], "action": f["action"], "text": f["text"]}
            for f in conn.execute(
                "SELECT * FROM feedback WHERE story_id = ? AND ep_no BETWEEN ? AND ? AND text IS NOT NULL ORDER BY id", span)
        ],
    }


def save_summary(conn: sqlite3.Connection, story_id: int, start: int, end: int, narrative: str,
                 snapshot: dict[str, Any]) -> int:
    return conn.execute(
        "INSERT INTO summaries (story_id, level, start_ep, end_ep, narrative, snapshot) VALUES (?, 'arc', ?, ?, ?, ?)",
        (story_id, start, end, narrative, to_json(snapshot)),
    ).lastrowid
