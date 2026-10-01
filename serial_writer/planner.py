"""The planning steps: story rules, acts, arcs, plan lines.

Each step only fills in what is missing from the current plan version. That one
rule gives us three things:
- safe to re-run: after a crash, finished parts are not made again;
- redo: leave a part out of a new version and the step rebuilds just that part;
- order: later parts are always built seeing the parts before them.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from . import plan_prompts as P
from .db import to_json, transaction
from .llm import CallContext, LLMClient, LLMError
from .plan_models import Bible, acts_model, arcs_model, beats_model, repetition_model, thread_key
from .plan_shape import Span, act_spans, all_arc_spans
from .plan_store import (
    available, latest_bible_id, latest_draft_plan_id, load_acts, load_arcs, load_beats,
    load_bible, load_cast, load_threads, new_plan_version, save_bible, seed_from_bible,
    thread_lifecycle, thread_status, upsert_character,
)
from .similarity import REPEAT, near_copies, overlap
from .story import get_story
from .tracing import log_step


class Planner:
    def __init__(self, conn: sqlite3.Connection, llm: LLMClient) -> None:
        self.conn = conn
        self.llm = llm

    # ------------------------------------------------------------ setup

    def setup(
        self, story_id: int, plan_version_id: int | None, redo: dict[str, Any] | None, run_id: int | None
    ) -> int:
        """Make sure there are story rules and a draft plan to fill. Returns the plan version id."""
        story = get_story(self.conn, story_id)
        bible_id = latest_bible_id(self.conn, story_id)

        if redo and redo["target"] == "bible":
            old_bible = self.conn.execute(
                "SELECT bible_version_id FROM plan_versions WHERE id = ?", (plan_version_id,)
            ).fetchone()["bible_version_id"]
            # If the new rules were already made before a crash, don't make them twice.
            if bible_id == old_bible:
                bible_id = self._make_bible(story, redo.get("note"), run_id, reason="redo")
            with transaction(self.conn):
                pv = new_plan_version(
                    self.conn, story_id, bible_id, created_by="model",
                    reason=redo.get("note") or "rebuild from new story rules", parent_id=plan_version_id,
                )
                seed_from_bible(self.conn, pv, story_id, load_bible(self.conn, bible_id))
            return pv

        if bible_id is None:
            bible_id = self._make_bible(story, None, run_id, reason="first draft")
        if plan_version_id is not None:
            return plan_version_id
        existing = latest_draft_plan_id(self.conn, story_id)
        if existing is not None:
            return existing
        with transaction(self.conn):
            pv = new_plan_version(self.conn, story_id, bible_id, created_by="model", reason="first draft")
            seed_from_bible(self.conn, pv, story_id, load_bible(self.conn, bible_id))
        return pv

    def _make_bible(self, story: sqlite3.Row, note: str | None, run_id: int | None, reason: str) -> int:
        bible = self.llm.structured(
            P.bible_prompt(story["premise"], story["total_episodes"], note), Bible,
            self._ctx("bible", story["id"], run_id), temperature=0.7, max_tokens=5000,
        )
        data = _clean_bible(bible.model_dump(), story["total_episodes"])
        with transaction(self.conn):
            bible_id = save_bible(self.conn, story["id"], data, created_by="model", reason=note or reason)
        log_step(self.conn, "bible", "created", run_id=run_id, story_id=story["id"],
                 detail={"bible_version_id": bible_id, "cast": len(data["cast"]), "threads": len(data["threads"])})
        return bible_id

    # ------------------------------------------------------------ acts

    def fill_acts(self, story_id: int, pv: int, note: str | None, run_id: int | None) -> None:
        if load_acts(self.conn, pv):
            return
        story = get_story(self.conn, story_id)
        bible = self._bible_for(pv)
        spans = act_spans(story["total_episodes"])
        model = acts_model(
            len(spans), [c["name"] for c in bible["cast"]], [t["key"] for t in bible["threads"]]
        )
        out = self.llm.structured(
            P.acts_prompt(story["premise"], bible, spans, note), model,
            self._ctx("plan_acts", story_id, run_id), temperature=0.7, max_tokens=5000,
        )
        with transaction(self.conn):
            for span, act in zip(spans, out.acts):
                details = {
                    "reveals": act.reveals,
                    "opens": list(act.opens),
                    "resolves": list(act.resolves),
                    "character_endpoints": [e.model_dump() for e in act.character_endpoints],
                }
                self.conn.execute(
                    """INSERT INTO plan_acts (plan_version_id, act_no, title, goal, turning_point, start_ep, end_ep, details)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pv, span.no, act.title, act.goal, act.turning_point, span.start, span.end, to_json(details)),
                )
        log_step(self.conn, "plan_acts", "created", run_id=run_id, story_id=story_id,
                 detail={"plan_version_id": pv, "acts": len(spans)})

    # ------------------------------------------------------------ arcs

    def fill_arcs(self, story_id: int, pv: int, note: str | None, run_id: int | None) -> None:
        acts = load_acts(self.conn, pv)
        spans_by_act = all_arc_spans([Span(a["act_no"], a["start_ep"], a["end_ep"]) for a in acts])
        bible = self._bible_for(pv)
        for act in acts:
            arcs = load_arcs(self.conn, pv)
            if any(a["act_no"] == act["act_no"] for a in arcs):
                continue
            spans = spans_by_act[act["act_no"]]
            names = self._names_available(pv, bible, spans[0].no - 1, act["end_ep"])
            out = self.llm.structured(
                P.arcs_prompt(
                    bible, acts, act["act_no"], spans,
                    [a for a in arcs if a["act_no"] < act["act_no"]],
                    [a for a in arcs if a["act_no"] > act["act_no"]],
                    [t for t in load_threads(self.conn, pv) if t["source"] != "bible"],
                    [c["name"] for c in load_cast(self.conn, pv) if c["source"] != "bible"],
                    note,
                ),
                arcs_model(len(spans), names), self._ctx("plan_arcs", story_id, run_id),
                temperature=0.7, max_tokens=5000,
            )
            with transaction(self.conn):
                for span, arc in zip(spans, out.arcs):
                    added = self._add_arc_extras(story_id, pv, span.no, arc, run_id)
                    details = {"focus_characters": list(arc.focus_characters), **added}
                    self.conn.execute(
                        """INSERT INTO plan_arcs (plan_version_id, arc_no, act_no, title, goal, turning_point,
                               start_ep, end_ep, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (pv, span.no, act["act_no"], arc.title, arc.goal, arc.turning_point,
                         span.start, span.end, to_json(details)),
                    )
            log_step(self.conn, "plan_arcs", "created", run_id=run_id, story_id=story_id,
                     detail={"plan_version_id": pv, "act": act["act_no"], "arcs": len(spans)})

    def _add_arc_extras(self, story_id: int, pv: int, arc_no: int, arc: Any, run_id: int | None) -> dict[str, list[str]]:
        """Register an arc's new subplots and people. Clashing names are skipped and logged."""
        source = f"arc:{arc_no}"
        existing = load_threads(self.conn, pv)
        keys = {t["key"] for t in existing}
        titles = {_plain(t["title"]) for t in existing}
        new_keys = []
        for t in arc.new_threads:
            key = thread_key(t.key or t.title)
            # The model likes to "add" a thread that already exists; that only
            # splits one question across two keys.
            if key in keys or _plain(t.title) in titles:
                log_step(self.conn, "plan_arcs", "skipped_new_thread", run_id=run_id, story_id=story_id,
                         detail={"arc": arc_no, "key": key, "title": t.title, "why": "already exists"})
                continue
            keys.add(key)
            titles.add(_plain(t.title))
            self.conn.execute(
                "INSERT INTO plan_threads (plan_version_id, key, title, question, source) VALUES (?, ?, ?, ?, ?)",
                (pv, key, t.title, t.question, source),
            )
            new_keys.append(key)

        cast = [_plain(c["name"]) for c in load_cast(self.conn, pv)]
        new_names = []
        for c in arc.new_characters:
            name = c.name.strip()
            plain = _plain(name)
            # Also catches variants of an existing person, like "Sarah Vance (Voice Only)".
            # Whole words only, so "Al" doesn't match "Alex".
            if not plain or any(f" {n} " in f" {plain} " or f" {plain} " in f" {n} " for n in cast):
                log_step(self.conn, "plan_arcs", "skipped_new_character", run_id=run_id, story_id=story_id,
                         detail={"arc": arc_no, "name": name, "why": "empty or already in cast"})
                continue
            cast.append(plain)
            char_id = upsert_character(self.conn, story_id, name, c.role, c.description, c.importance)
            self.conn.execute(
                "INSERT INTO plan_character_arcs (plan_version_id, character_id, arc, source) VALUES (?, ?, ?, ?)",
                (pv, char_id, f"Wants: {c.wants} Secret: {c.secret}", source),
            )
            new_names.append(name)
        return {"new_threads": new_keys, "new_characters": new_names}

    # ------------------------------------------------------------ plan lines

    def fill_beats(self, story_id: int, pv: int, note: str | None, run_id: int | None) -> None:
        acts = {a["act_no"]: a for a in load_acts(self.conn, pv)}
        arcs = load_arcs(self.conn, pv)
        bible = self._bible_for(pv)
        for i, arc in enumerate(arcs):
            if load_beats(self.conn, pv, arc["start_ep"], arc["end_ep"]):
                continue
            self._plan_arc_beats(story_id, pv, bible, acts, arcs, i, note, run_id)

    def _plan_arc_beats(
        self, story_id: int, pv: int, bible: dict[str, Any], acts: dict[int, dict[str, Any]],
        arcs: list[dict[str, Any]], i: int, note: str | None, run_id: int | None,
    ) -> None:
        """Write one arc's plan lines, check them, repair once if needed, then save."""
        arc = arcs[i]
        act = acts[arc["act_no"]]
        prev_arc = arcs[i - 1] if i > 0 else None
        next_arc = arcs[i + 1] if i + 1 < len(arcs) else None
        eps = list(range(arc["start_ep"], arc["end_ep"] + 1))

        threads = [t for t in load_threads(self.conn, pv) if available(t["source"], arc["arc_no"])]
        earlier = load_beats(self.conn, pv, 1, arc["start_ep"] - 1)
        status = thread_status(threads, earlier, arc["start_ep"])
        state_before = {t["key"]: t["state"] for t in status}
        # Code decides what the model may pick: no resolved threads, nobody before they arrive.
        usable_keys = [t["key"] for t in status if t["state"] != "resolved"]
        names = self._names_available(pv, bible, arc["arc_no"], arc["end_ep"])
        must_close = self._must_close(status, act, arc, next_arc)

        prompt = P.beats_prompt(
            bible, act, arc, prev_arc,
            load_beats(self.conn, pv, prev_arc["start_ep"], prev_arc["end_ep"]) if prev_arc else [],
            next_arc, P.render_thread_status(status, arc["start_ep"]),
            [t for t in threads if t["source"] != "bible"], must_close, note,
        )
        model = beats_model(len(eps), names, usable_keys)
        ctx = self._ctx("plan_beats", story_id, run_id)
        out = self.llm.structured(prompt, model, ctx, temperature=0.7, max_tokens=6000)

        hard = self._beat_problems(out.beats, eps, state_before, must_close, earlier)
        copied = {int(p.split()[1]) for p in hard if "nearly a copy" in p}
        problems = hard + self._repeat_problems(story_id, earlier, out.beats, eps, run_id, skip=copied)
        if problems:
            log_step(self.conn, "plan_beats", "repair", run_id=run_id, story_id=story_id,
                     detail={"arc": arc["arc_no"], "problems": problems})
            repaired = self.llm.structured(
                P.repair_messages(prompt, out.model_dump_json(), problems), model,
                self._ctx("plan_beats_repair", story_id, run_id), temperature=0.5, max_tokens=6000,
            )
            left = self._beat_problems(repaired.beats, eps, state_before, must_close, earlier)
            # Keep the rewrite unless it made the code-checked problems worse.
            if len(left) <= len(hard):
                out = repaired
            else:
                left = hard
            if left:
                log_step(self.conn, "plan_beats", "unfixed_after_repair", run_id=run_id,
                         story_id=story_id, detail={"arc": arc["arc_no"], "problems": left})

        with transaction(self.conn):
            for ep, beat in zip(eps, out.beats):
                self.conn.execute(
                    """INSERT INTO plan_beats (plan_version_id, ep_no, beat, hook, characters, threads, is_turning_point)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (pv, ep, beat.beat, beat.hook, to_json(list(dict.fromkeys(beat.characters))),
                     to_json([m.model_dump() for m in beat.threads]), int(ep == act["end_ep"])),
                )
        log_step(self.conn, "plan_beats", "created", run_id=run_id, story_id=story_id,
                 detail={"plan_version_id": pv, "arc": arc["arc_no"], "episodes": len(eps),
                         "repaired": bool(problems)})

    def _names_available(self, pv: int, bible: dict[str, Any], arc_no: int, by_ep: int) -> list[str]:
        """People in the story by episode `by_ep`: bible cast by their arrival
        episode, arc-added people from their arc on."""
        arrives = {c["name"]: c["enters_around_ep"] for c in bible["cast"]}
        names = [
            c["name"] for c in load_cast(self.conn, pv)
            if available(c["source"], arc_no) and arrives.get(c["name"], 1) <= by_ep
        ]
        # Never leave the model with nobody to pick.
        return names or [c["name"] for c in bible["cast"] if c["importance"] == "major"][:1]

    @staticmethod
    def _must_close(
        status: list[dict[str, Any]], act: dict[str, Any], arc: dict[str, Any], next_arc: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Threads this arc has to resolve: everything still open in the story's last
        arc, and what the act promised to resolve, in the act's last arc."""
        still_open = [t for t in status if t["state"] == "open"]
        if next_arc is None:
            return still_open
        if arc["end_ep"] == act["end_ep"]:
            promised = set(act["details"].get("resolves", []))
            return [t for t in still_open if t["key"] in promised]
        return []

    @staticmethod
    def _beat_problems(
        beats: list[Any], eps: list[int], state_before: dict[str, str], must_close: list[dict[str, Any]],
        earlier: list[dict[str, Any]],
    ) -> list[str]:
        """What code can tell for sure is wrong with an arc's plan lines."""
        closing = {t["key"] for t in must_close}
        # Copies of earlier lines (the model sometimes copies the previous arc it was shown).
        out = [
            f"Ep {ep} is nearly a copy of ep {src} ({int(score * 100)}% the same words). Write a new event."
            for ep, src, score in near_copies(
                [(ep, b.beat) for ep, b in zip(eps, beats)], [(b["ep_no"], b["beat"]) for b in earlier]
            )
        ]
        for key, state in state_before.items():
            moves = [(ep, m.event) for ep, b in zip(eps, beats) for m in b.threads if m.key == key]
            found, final = thread_lifecycle(moves, state)
            out += [f"Ep {ep}: thread [{key}] {msg}" for ep, _, msg in found]
            if key in closing and final != "resolved":
                out.append(f"Thread [{key}] must be resolved (answered for good) in this arc, but isn't")
        return out

    def _repeat_problems(
        self, story_id: int, earlier: list[dict[str, Any]], beats: list[Any], eps: list[int], run_id: int | None,
        skip: set[int] = frozenset(),
    ) -> list[str]:
        """Ask the model, one row per new episode, which earlier episode is most like it.
        The model finds the pair; code decides if it's a repeat (the model's own yes/no
        is unreliable: it has matched copies correctly and still said "not the same")."""
        new = [{"ep_no": ep, "beat": b.beat, "hook": b.hook} for ep, b in zip(eps, beats)]
        try:
            out = self.llm.structured(
                P.repetition_prompt(earlier, new), repetition_model(eps, eps[-1]),
                self._ctx("plan_repeat_check", story_id, run_id), temperature=0.0, max_tokens=3000,
            )
        except LLMError as exc:
            log_step(self.conn, "plan_repeat_check", "failed", run_id=run_id, story_id=story_id,
                     detail={"error": str(exc)})
            return []
        text = {b["ep_no"]: b["beat"] for b in earlier} | {ep: b.beat for ep, b in zip(eps, beats)}
        problems = []
        for r in out.rows:
            if r.ep_no in skip or r.closest_ep not in text or r.closest_ep == r.ep_no:
                continue
            if r.same_event or overlap(text[r.ep_no], text[r.closest_ep]) >= REPEAT:
                problems.append(f"Ep {r.ep_no} repeats ep {r.closest_ep} ({r.what_is_alike}). Give it a different event.")
        return problems

    # ------------------------------------------------------------ helpers

    def _bible_for(self, pv: int) -> dict[str, Any]:
        row = self.conn.execute("SELECT bible_version_id FROM plan_versions WHERE id = ?", (pv,)).fetchone()
        return load_bible(self.conn, row["bible_version_id"])

    @staticmethod
    def _ctx(node: str, story_id: int, run_id: int | None) -> CallContext:
        return CallContext(node=node, story_id=story_id, run_id=run_id)


def _plain(text: str) -> str:
    """Lowercase words only, for comparing names and titles: "The Building's Lock" -> "the building s lock"."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _clean_bible(data: dict[str, Any], total_episodes: int) -> dict[str, Any]:
    """Code-side clean-up of the model's story rules: unique names and keys, sane episode numbers."""
    seen: set[str] = set()
    cast = []
    for c in data["cast"]:
        name = c["name"].strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        cast.append({**c, "name": name, "enters_around_ep": min(max(1, c["enters_around_ep"]), total_episodes)})
    keys: set[str] = set()
    threads = []
    for t in data["threads"]:
        key = base = thread_key(t["key"] or t["title"])
        n = 2
        while key in keys:
            key, n = f"{base}_{n}", n + 1
        keys.add(key)
        threads.append({**t, "key": key})
    return {**data, "cast": cast, "threads": threads}
