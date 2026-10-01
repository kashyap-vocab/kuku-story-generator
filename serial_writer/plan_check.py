"""Checking a finished plan before the reviewer sees it.

Code checks what code can know for sure (coverage, thread order, missing
people). The model reviews each act for repetition, pacing and logic, and must
quote the plan line it is complaining about; complaints whose quote isn't
really there are dropped. Nothing is fixed silently: problems go to the reviewer.
"""

from __future__ import annotations

import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import plan_prompts as P
from .db import to_json
from .llm import CallContext, LLMClient, LLMError
from .plan_models import act_review_model, rules_review_model
from .plan_store import (
    load_acts, load_arcs, load_beats, load_bible, load_cast, load_threads, thread_lifecycle,
)
from .similarity import near_copies
from .tracing import log_step

# A thread or main character quiet for longer than this gets flagged.
THREAD_QUIET_EPS = 30
MAJOR_ABSENT_EPS = 40
MAX_PARALLEL = 5


def problem(kind: str, eps: list[int], message: str, severity: str, source: str = "code", **extra: Any) -> dict[str, Any]:
    return {"source": source, "kind": kind, "eps": eps, "message": message, "severity": severity, **extra}


def code_checks(conn: sqlite3.Connection, pv: int, total: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    beats = load_beats(conn, pv)

    # Every episode planned exactly once, and inside an arc and an act.
    eps = [b["ep_no"] for b in beats]
    missing = sorted(set(range(1, total + 1)) - set(eps))
    if missing:
        out.append(problem("coverage", missing[:20], f"{len(missing)} episodes have no plan line", "must_fix"))
    for spans, label in ((load_acts(conn, pv), "act"), (load_arcs(conn, pv), "arc")):
        covered = [e for s in spans for e in range(s["start_ep"], s["end_ep"] + 1)]
        if sorted(covered) != list(range(1, total + 1)):
            out.append(problem("coverage", [], f"{label}s don't cover episodes 1-{total} exactly once", "must_fix"))

    # Plan lines only use people and threads the plan knows about. (A redo can
    # remove an arc's new characters or subplots that later lines still use.)
    names = {c["name"] for c in load_cast(conn, pv)}
    keys = {t["key"] for t in load_threads(conn, pv)}
    for b in beats:
        for n in b["characters"]:
            if n not in names:
                out.append(problem("character", [b["ep_no"]], f"{n} is not in the cast", "must_fix"))
        for m in b["threads"]:
            if m["key"] not in keys:
                out.append(problem("thread", [b["ep_no"]], f"Thread [{m['key']}] does not exist", "must_fix", key=m["key"]))

    # Copied plan lines.
    for ep, src, score in near_copies([(b["ep_no"], b["beat"]) for b in beats], []):
        out.append(problem("repetition", [src, ep], f"Ep {ep} is nearly a copy of ep {src} "
                           f"({int(score * 100)}% the same words)", "must_fix"))

    # Threads: opened before they move, nothing after resolving, not left quiet too long.
    for t in load_threads(conn, pv):
        moves = [(b["ep_no"], m["event"]) for b in beats for m in b["threads"] if m["key"] == t["key"]]
        name = f"'{t['title']}' [{t['key']}]"
        if not moves:
            out.append(problem("thread", [], f"Thread {name} is never used", "minor", key=t["key"]))
            continue
        found, state = thread_lifecycle(moves)
        out += [problem("thread", [ep], f"Thread {name} {msg}", sev, key=t["key"]) for ep, sev, msg in found]
        # Long silences while the thread is still open (up to its first resolve).
        resolved_at = next((i for i, (_, e) in enumerate(moves) if e == "resolve"), len(moves) - 1)
        live = moves[: resolved_at + 1]
        for (a, _), (b, _) in zip(live, live[1:]):
            if b - a > THREAD_QUIET_EPS:
                out.append(problem("thread", [a, b], f"Thread {name} goes quiet for {b - a} episodes", "minor", key=t["key"]))
        last = moves[-1][0]
        if state == "open":
            if total - last > THREAD_QUIET_EPS:
                out.append(problem("thread", [last], f"Thread {name} is dropped after ep {last}", "minor", key=t["key"]))
            else:
                out.append(problem("thread", [last], f"Thread {name} is still open at the end of the story", "minor", key=t["key"]))

    # Major characters: they show up, and don't vanish for too long.
    for c in load_cast(conn, pv):
        if c["importance"] != "major":
            continue
        seen = [b["ep_no"] for b in beats if c["name"] in b["characters"]]
        if not seen:
            out.append(problem("character", [], f"Major character {c['name']} never appears", "must_fix"))
            continue
        for a, b in zip(seen, seen[1:]):
            if b - a > MAJOR_ABSENT_EPS:
                out.append(problem("character", [a, b], f"{c['name']} disappears for {b - a} episodes", "minor"))
    return out


# How an act-review issue counts. Logic and character breaks must be fixed;
# pacing and hooks are the reviewer's call.
ISSUE_SEVERITY = {"logic": "must_fix", "character": "must_fix", "pacing": "minor", "hook": "minor"}


def model_checks(
    conn: sqlite3.Connection, llm: LLMClient, story_id: int, pv: int, run_id: int | None
) -> tuple[list[dict[str, Any]], int]:
    """A check of the story rules plus one review per act, all in parallel.
    Returns (verified problems, number dropped as unverifiable)."""
    bible_id = conn.execute("SELECT bible_version_id FROM plan_versions WHERE id = ?", (pv,)).fetchone()[0]
    bible = load_bible(conn, bible_id)
    acts = load_acts(conn, pv)
    beats = load_beats(conn, pv)

    def review_act(act: dict[str, Any]) -> Any:
        mine = [b for b in beats if act["start_ep"] <= b["ep_no"] <= act["end_ep"]]
        earlier = [b for b in beats if b["ep_no"] < act["start_ep"]]
        return llm.structured(
            P.act_review_prompt(bible, act, mine, earlier),
            act_review_model([b["ep_no"] for b in mine]),
            CallContext(node="plan_check", story_id=story_id, run_id=run_id),
            temperature=0.0, max_tokens=6000,
        )

    items = P.rules_items(bible)

    def review_rules() -> Any:
        return llm.structured(
            P.rules_review_prompt(bible), rules_review_model([i for i, _ in items]),
            CallContext(node="rules_check", story_id=story_id, run_id=run_id),
            temperature=0.0, max_tokens=4000,
        )

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(acts) + 1)) as pool:
        rules_future = pool.submit(review_rules)
        futures = [(act, pool.submit(review_act, act)) for act in acts]

    found: list[dict[str, Any]] = []
    dropped = 0

    # Story rules: a clash only counts if the quoted words really are in the rules,
    # and aren't just the item quoting itself.
    rules_text = P.render_bible(bible)
    item_text = dict(items)
    try:
        for row in rules_future.result().rows:
            clash = row.conflicts_with.strip()
            if not clash or _norm(clash) in ("none", "n/a", ""):
                continue
            if _contains(rules_text, clash) and not _contains(item_text[row.item], clash):
                found.append(problem("rules", [], f"[{row.item}] {item_text[row.item]} -- clashes with: "
                                     f"\"{clash}\". {row.note}", "must_fix", source="model", quote=clash))
            else:
                dropped += 1
                log_step(conn, "rules_check", "dropped_unverified", run_id=run_id, story_id=story_id,
                         detail=row.model_dump())
    except LLMError as exc:
        found.append(problem("review_failed", [], f"Story rules could not be reviewed: {exc}", "minor"))

    for act, fut in futures:
        try:
            out = fut.result()
        except LLMError as exc:
            found.append(problem("review_failed", [], f"Act {act['act_no']} could not be reviewed: {exc}", "minor"))
            continue
        for row in out.rows:
            if row.issue == "none":
                continue
            found.append(problem(row.issue, [row.ep_no], row.note or row.what_changes,
                                 ISSUE_SEVERITY[row.issue], source="model"))
    return found, dropped


def run_plan_check(
    conn: sqlite3.Connection, llm: LLMClient, story_id: int, pv: int, total: int, run_id: int | None
) -> dict[str, Any]:
    code = code_checks(conn, pv, total)
    model, dropped = model_checks(conn, llm, story_id, pv, run_id)
    report = {"problems": code + model, "dropped_unverified": dropped}
    conn.execute("UPDATE plan_versions SET check_report = ? WHERE id = ?", (to_json(report), pv))
    log_step(conn, "plan_check", "checked", run_id=run_id, story_id=story_id,
             detail={"plan_version_id": pv, "code": len(code), "model": len(model), "dropped": dropped})
    return report


def _norm(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip(" \"'.,;:!?")


def _contains(haystack: str, quote: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(haystack)
