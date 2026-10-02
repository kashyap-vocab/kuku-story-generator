"""Applying the reviewer's decision on a plan.

Every decision is stored in `feedback` (which can never be changed or deleted).
Edits and redos make a new plan version; the old one stays as history.
"""

from __future__ import annotations

import copy
import sqlite3
from typing import Any

from .db import to_json, transaction
from .plan_models import PlanChanges, PlanDecision
from .plan_store import (
    copy_plan, get_plan_version, load_acts, load_arcs, load_beats, load_bible, load_cast,
    load_threads, new_plan_version, save_bible,
)
from .story import set_status


class PlanDecisionError(ValueError):
    """The decision can't be applied as sent (unknown act, unknown character...)."""


def apply_plan_decision(
    conn: sqlite3.Connection, story_id: int, pv: int, raw: dict[str, Any]
) -> dict[str, Any]:
    """Returns {"outcome": "approved" | "edited" | "redo", "plan_version_id": ..., "redo": ...}."""
    try:
        decision = PlanDecision.model_validate(raw)
    except Exception as exc:
        raise PlanDecisionError(f"decision not understood: {exc}") from exc

    if decision.action == "approve":
        _approve(conn, story_id, pv, decision)
        return {"outcome": "approved", "plan_version_id": pv, "redo": None}
    if decision.action == "edit":
        if decision.changes is None:
            raise PlanDecisionError("edit needs 'changes'")
        new_pv = _edit(conn, story_id, pv, decision.changes, decision.note)
        return {"outcome": "edited", "plan_version_id": new_pv, "redo": None}
    return _redo(conn, story_id, pv, decision)


def _approve(conn: sqlite3.Connection, story_id: int, pv: int, d: PlanDecision) -> None:
    plan = get_plan_version(conn, pv)
    with transaction(conn):
        _feedback(conn, story_id, pv, "plan_approve", d.note, None)
        conn.execute(
            "UPDATE bible_versions SET status = 'superseded' WHERE story_id = ? AND status = 'approved' AND id <> ?",
            (story_id, plan["bible_version_id"]),
        )
        conn.execute("UPDATE bible_versions SET status = 'approved' WHERE id = ?", (plan["bible_version_id"],))
        conn.execute(
            "UPDATE plan_versions SET status = 'superseded' WHERE story_id = ? AND status = 'approved'",
            (story_id,),
        )
        conn.execute("UPDATE plan_versions SET status = 'approved' WHERE id = ?", (pv,))
        set_status(conn, story_id, "writing")


def _edit(conn: sqlite3.Connection, story_id: int, pv: int, ch: PlanChanges, note: str | None) -> int:
    plan = get_plan_version(conn, pv)
    acts = {a["act_no"] for a in load_acts(conn, pv)}
    arcs = {a["arc_no"] for a in load_arcs(conn, pv)}
    eps = {b["ep_no"] for b in load_beats(conn, pv)}
    names = {c["name"] for c in load_cast(conn, pv)}
    keys = {t["key"] for t in load_threads(conn, pv)}

    bad = [f"act {n}" for n in ch.acts if n not in acts] + [f"arc {n}" for n in ch.arcs if n not in arcs]
    bad += [f"episode {n}" for n in ch.beats if n not in eps]
    for ep, b in ch.beats.items():
        bad += [f"character '{n}' (ep {ep})" for n in (b.characters or []) if n not in names]
        bad += [f"thread '{m.key}' (ep {ep})" for m in (b.threads or []) if m.key not in keys]
    if ch.bible:
        bad += [f"character '{n}'" for n in ch.bible.cast if n not in names]
    if bad:
        raise PlanDecisionError("unknown " + ", ".join(bad))

    with transaction(conn):
        bible_id = plan["bible_version_id"]
        if ch.bible and ch.bible.model_dump(exclude_none=True, exclude_defaults=True):
            bible = _edited_bible(load_bible(conn, bible_id), ch)
            bible_id = save_bible(conn, story_id, bible, created_by="human", reason=note or "edited by reviewer")
        new_pv = new_plan_version(conn, story_id, bible_id, created_by="human",
                                  reason=note or "edited by reviewer", parent_id=pv)
        copy_plan(conn, pv, new_pv)

        if ch.bible:
            for name, c in ch.bible.cast.items():
                if c.arc is not None:
                    conn.execute(
                        """UPDATE plan_character_arcs SET arc = ? WHERE plan_version_id = ? AND character_id =
                               (SELECT id FROM characters WHERE story_id = ? AND name = ?)""",
                        (c.arc, new_pv, story_id, name),
                    )
        for no, e in ch.acts.items():
            _update(conn, "plan_acts", "act_no", new_pv, no, e.model_dump(exclude_none=True))
            conn.execute(
                """UPDATE plan_arcs SET title = a.title, goal = a.goal, turning_point = a.turning_point
                   FROM plan_acts a WHERE plan_arcs.plan_version_id = ? AND a.plan_version_id = ? AND a.act_no = ?
                   AND plan_arcs.act_no = a.act_no AND plan_arcs.start_ep = a.start_ep AND plan_arcs.end_ep = a.end_ep""",
                (new_pv, new_pv, no),
            )
        for no, e in ch.arcs.items():
            _update(conn, "plan_arcs", "arc_no", new_pv, no, e.model_dump(exclude_none=True))
        for ep, e in ch.beats.items():
            fields = e.model_dump(exclude_none=True)
            if "characters" in fields:
                fields["characters"] = to_json(fields["characters"])
            if "threads" in fields:
                fields["threads"] = to_json(fields["threads"])
            _update(conn, "plan_beats", "ep_no", new_pv, ep, fields)

        _feedback(conn, story_id, pv, "plan_edit", note, ch.model_dump(exclude_none=True))
    return new_pv


def _edited_bible(bible: dict[str, Any], ch: PlanChanges) -> dict[str, Any]:
    out = copy.deepcopy(bible)
    edits = ch.bible.model_dump(exclude_none=True)
    cast_edits = edits.pop("cast", {})
    out.update(edits)
    for c in out["cast"]:
        if c["name"] in cast_edits:
            c.update(cast_edits[c["name"]])
    return out


_EDITABLE = {
    "plan_acts": {"title", "goal", "turning_point"},
    "plan_arcs": {"title", "goal", "turning_point"},
    "plan_beats": {"beat", "hook", "characters", "threads"},
}


def _update(conn: sqlite3.Connection, table: str, key_col: str, pv: int, key: int, fields: dict[str, Any]) -> None:
    fields = {k: v for k, v in fields.items() if k in _EDITABLE[table]}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE {table} SET {sets} WHERE plan_version_id = ? AND {key_col} = ?",
        (*fields.values(), pv, key),
    )


def _redo(conn: sqlite3.Connection, story_id: int, pv: int, d: PlanDecision) -> dict[str, Any]:
    if d.target is None:
        raise PlanDecisionError("redo needs a 'target': bible, all, act or arc")
    if d.target in ("act", "arc") and d.no is None:
        raise PlanDecisionError(f"redo of an {d.target} needs its number in 'no'")
    if d.target == "act" and d.no not in {a["act_no"] for a in load_acts(conn, pv)}:
        raise PlanDecisionError(f"there is no act {d.no}")
    if d.target == "arc" and d.no not in {a["arc_no"] for a in load_arcs(conn, pv)}:
        raise PlanDecisionError(f"there is no arc {d.no}")

    redo = {"target": d.target, "no": d.no, "note": d.note}
    plan = get_plan_version(conn, pv)
    with transaction(conn):
        _feedback(conn, story_id, pv, "plan_redo", d.note, redo)
        if d.target == "bible":
            set_status(conn, story_id, "planning")
            return {"outcome": "redo", "plan_version_id": pv, "redo": redo}
        reason = d.note or f"redo {d.target} {d.no or ''}".strip()
        new_pv = new_plan_version(conn, story_id, plan["bible_version_id"], created_by="model",
                                  reason=reason, parent_id=pv)
        if d.target == "all":
            copy_plan(conn, pv, new_pv, keep_acts=False)
        elif d.target == "act":
            copy_plan(conn, pv, new_pv, skip_arcs_of_acts={d.no})
        else:
            copy_plan(conn, pv, new_pv, skip_beats_of_arcs={d.no})
        set_status(conn, story_id, "planning")
    return {"outcome": "redo", "plan_version_id": new_pv, "redo": redo}


def _feedback(
    conn: sqlite3.Connection, story_id: int, pv: int, action: str, text: str | None, detail: Any
) -> None:
    conn.execute(
        "INSERT INTO feedback (story_id, plan_version_id, action, text, classification) VALUES (?, ?, ?, ?, ?)",
        (story_id, pv, action, text, to_json(detail) if detail is not None else None),
    )
