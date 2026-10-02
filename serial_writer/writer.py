"""Writing one episode: outline, draft, check, revise, remember. And applying
the human's decision on it, including feedback that carries forward.

Each method is one graph step and is safe to re-run: it works from what is in
story.db, and the graph state only carries ids.
"""

from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from . import episode_prompts as E
from .db import to_json, transaction
from .episode_models import ContinuityOut, FeedbackSort, extraction_model, outline_model, plan_check_model
from .llm import BudgetExceeded, CallContext, LLMClient, LLMError
from .memory import (
    approve_version, approved_episodes, arc_snapshot, arcs_to_summarize, build_pack, character_states, directives,
    save_context, save_memory, save_summary,
)
from .plan_models import beats_model
from .plan_prompts import render_bible
from .plan_store import (
    copy_plan, get_plan_version, load_arcs, load_beats, load_bible, load_cast, load_threads, new_plan_version,
)
from .similarity import REPEAT, contains_quote, overlap
from .story import current_review_setting
from .tracing import log_step


class EpisodeDecisionError(ValueError):
    """The reviewer's decision can't be applied as sent."""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


class Writer:
    def __init__(self, conn: sqlite3.Connection, llm: LLMClient) -> None:
        self.conn = conn
        self.llm = llm


    def draft(self, story_id: int, ep: int, pv: int, note: str | None, since: str, run_id: int | None) -> int:
        """Build the memory pack, outline, draft. Returns the new version id."""
        pack, refs = build_pack(self.conn, story_id, ep, pv)
        names = [c["name"] for c in self._cast(story_id, pv)]
        ctx = self._ctx("outline", story_id, ep, run_id, since)
        outline = self.llm.structured(E.outline_prompt(pack, ep, note), outline_model(names), ctx,
                                      temperature=0.7, max_tokens=1500).model_dump()
        text = self.llm.complete(E.draft_prompt(pack, ep, outline, note),
                                 self._ctx("draft", story_id, ep, run_id, since),
                                 temperature=0.8, max_tokens=1600).text.strip()
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE episode_versions SET status = 'superseded' WHERE story_id = ? AND ep_no = ? AND status IN ('draft', 'in_review')",
                (story_id, ep),
            )
            context_id = save_context(self.conn, story_id, ep, pv, pack, refs)
            vid = self._new_version(story_id, ep, pv, text, "model", context_id=context_id, outline=outline)
        log_step(self.conn, "draft", "drafted", run_id=run_id, story_id=story_id, ep_no=ep,
                 detail={"version_id": vid, "words": word_count(text), "pack_tokens_est": len(pack) // 4,
                         "rewrite_note": note})
        return vid

    def _new_version(self, story_id: int, ep: int, pv: int, text: str, by: str, *, context_id: int | None,
                     outline: dict[str, Any] | None = None, parent_id: int | None = None, revisions: int = 0) -> int:
        version = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM episode_versions WHERE story_id = ? AND ep_no = ?", (story_id, ep)
        ).fetchone()[0]
        return self.conn.execute(
            """INSERT INTO episode_versions (story_id, ep_no, version, parent_id, plan_version_id, text, word_count,
                   created_by, status, revisions, context_id, outline) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?)""",
            (story_id, ep, version, parent_id, pv, text, word_count(text), by, revisions, context_id,
             to_json(outline) if outline else None),
        ).lastrowid


    def check(self, story_id: int, vid: int, since: str, run_id: int | None) -> dict[str, Any]:
        """Code checks, then the continuity and plan checks in parallel. Saves and returns the report."""
        v = self._version(vid)
        ep, text = v["ep_no"], v["text"]
        pack = self._pack(v)
        problems = self._code_checks(story_id, ep, text)
        dropped: list[dict[str, Any]] = []
        try:
            model_problems, dropped = self._model_checks(story_id, v, pack, since, run_id)
            problems += model_problems
        except BudgetExceeded as exc:
            problems.append(_problem("budget", f"Token budget reached: {exc}", "must_fix", source="code"))
        except LLMError as exc:
            problems.append(_problem("check_failed", f"A check could not run: {exc}", "minor", source="code"))
        report = {"problems": problems, "dropped_unverified": dropped}
        self.conn.execute("UPDATE episode_versions SET check_report = ? WHERE id = ?", (to_json(report), vid))
        log_step(self.conn, "check", "checked", run_id=run_id, story_id=story_id, ep_no=ep,
                 detail={"version_id": vid, "must_fix": sum(p["severity"] == "must_fix" for p in problems),
                         "minor": sum(p["severity"] == "minor" for p in problems), "dropped": len(dropped)})
        return report

    def _code_checks(self, story_id: int, ep: int, text: str) -> list[dict[str, Any]]:
        out = []
        words = word_count(text)
        if not E.MIN_WORDS <= words <= E.MAX_WORDS:
            out.append(_problem("length", f"{words} words; it must be {E.MIN_WORDS}-{E.MAX_WORDS}", "must_fix", source="code"))
        first = text.strip().splitlines()[0] if text.strip() else ""
        if not first.lower().startswith(f"episode {ep}"):
            out.append(_problem("format", f"First line should be 'Episode {ep}: <title>', got '{first[:60]}'", "minor", source="code"))
        low = text.casefold()
        for meta in ("in this episode", "to be continued", "end of episode", "word count"):
            if meta in low:
                out.append(_problem("format", f"Remove notes about the episode itself ('{meta}')", "must_fix",
                                    source="code", quote=meta))
        found = [c for c in E.CLICHES if c in low]
        if len(found) >= 2:
            out.append(_problem("cliche", "Stock phrases: " + "; ".join(found), "must_fix", source="code", quote=found[0]))
        elif found:
            out.append(_problem("cliche", f"Stock phrase: {found[0]}", "minor", source="code", quote=found[0]))
        for name, s in character_states(self.conn, story_id, ep).items():
            if s["status"] != "dead":
                continue
            for n in {name, name.split()[0]}:
                m = re.search(rf"\b{re.escape(n)}\s+(said|says|asked|asks|whispered|shouted|replied|yelled|muttered)\b"
                              rf"|\b(said|asked|whispered|shouted|replied)\s+{re.escape(n)}\b", text)
                if m:
                    out.append(_problem("dead_character", f"{name} is dead in the story's memory but speaks here",
                                        "minor", source="code", quote=m.group(0)))
                    break
        return out

    def _model_checks(
        self, story_id: int, v: sqlite3.Row, pack: str, since: str, run_id: int | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ep, text = v["ep_no"], v["text"]
        beat = load_beats(self.conn, v["plan_version_id"], ep, ep)[0]
        active = directives(self.conn, story_id)
        items = [("plan_line", f"Delivers this plan line: {beat['beat']}"),
                 ("hook", f"Ends on a strong hook (planned: {beat['hook']})")]
        items += [(f"instruction_{d['id']}", f"Follows the human instruction: {d['text']}") for d in active]
        earlier = self._related_episodes(story_id, ep, beat)

        with ThreadPoolExecutor(max_workers=2) as pool:
            cont_f = pool.submit(self.llm.structured, E.continuity_prompt(pack, text), ContinuityOut,
                                 self._ctx("check_continuity", story_id, ep, run_id, since), temperature=0.0, max_tokens=1500)
            plan_f = pool.submit(self.llm.structured, E.plan_check_prompt(pack, text, items, earlier),
                                 plan_check_model([i for i, _ in items]),
                                 self._ctx("check_plan", story_id, ep, run_id, since), temperature=0.0, max_tokens=1500)
            cont, plan = cont_f.result(), plan_f.result()

        problems: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for i in cont.issues:
            if contains_quote(text, i.quote) and contains_quote(pack, i.clashes_with):
                problems.append(_problem("continuity", f"{i.note} (memory says: \"{i.clashes_with}\")",
                                         "must_fix", quote=i.quote))
            else:
                dropped.append({"check": "continuity", **i.model_dump()})
        for r in plan.rows:
            if r.verdict == "yes":
                continue
            what = dict(items)[r.item]
            quoted = contains_quote(text, r.quote)
            if r.item.startswith("instruction_"):
                sev = "must_fix" if r.verdict == "no" and quoted else "minor"
            elif r.item == "plan_line":
                sev = "must_fix" if r.verdict == "no" else "minor"
            else:
                sev = "minor"
            problems.append(_problem(r.item.split("_")[0] if r.item.startswith("instruction") else r.item,
                                     f"{what} -- {r.verdict}: {r.note}", sev, quote=r.quote if quoted else None))
        if plan.same_event and plan.closest_earlier_ep:
            prev = next((e for e in earlier if e["ep_no"] == plan.closest_earlier_ep), None)
            if prev is not None:
                problems.append(_problem("repetition", f"Repeats ep {prev['ep_no']}: {plan.what_is_alike}", "minor"))
        if dropped:
            log_step(self.conn, "check", "dropped_unverified", run_id=run_id, story_id=story_id, ep_no=ep,
                     detail={"dropped": dropped})
        return problems, dropped

    def _related_episodes(self, story_id: int, ep: int, beat: dict[str, Any]) -> list[dict[str, Any]]:
        """Earlier episodes with the same people, or that read alike: what a repeat would repeat."""
        names = set(beat["characters"])
        done = approved_episodes(self.conn, story_id, ep)
        picked = []
        for e in done:
            plan = load_beats(self.conn, e["plan_version_id"], e["ep_no"], e["ep_no"])
            chars = set(plan[0]["characters"]) if plan else set()
            if names & chars or overlap(e["one_line"] or "", beat["beat"]) >= REPEAT / 2:
                picked.append(e)
        return picked[-15:]


    def revise(self, story_id: int, vid: int, since: str, run_id: int | None) -> int | None:
        """Rewrite to fix the must-fix problems. Returns the new version id, or None if
        the budget ran out (the episode then goes to the human as it is)."""
        v = self._version(vid)
        report = json.loads(v["check_report"])
        must = [p for p in report["problems"] if p["severity"] == "must_fix"]
        try:
            text = self.llm.complete(E.revise_prompt(self._pack(v), v["ep_no"], v["text"], must),
                                     self._ctx("revise", story_id, v["ep_no"], run_id, since),
                                     temperature=0.6, max_tokens=1600).text.strip()
        except BudgetExceeded:
            return None
        with transaction(self.conn):
            self.conn.execute("UPDATE episode_versions SET status = 'superseded' WHERE id = ?", (vid,))
            new = self._new_version(story_id, v["ep_no"], v["plan_version_id"], text, "model",
                                    context_id=v["context_id"], outline=json.loads(v["outline"]) if v["outline"] else None,
                                    parent_id=vid, revisions=v["revisions"] + 1)
        log_step(self.conn, "revise", "revised", run_id=run_id, story_id=story_id, ep_no=v["ep_no"],
                 detail={"from": vid, "to": new, "fixing": [p["message"] for p in must]})
        return new


    def extract(self, story_id: int, vid: int, run_id: int | None) -> dict[str, Any]:
        """Pull memory updates out of the final text and store them against this version."""
        v = self._version(vid)
        ep, pv = v["ep_no"], v["plan_version_id"]
        with transaction(self.conn):
            self._forget_version(vid)
        names = [c["name"] for c in self._cast(story_id, pv)]
        threads = load_threads(self.conn, pv)
        out = self.llm.structured(
            E.extraction_prompt(v["text"], ep, names, threads), extraction_model(names, [t["key"] for t in threads]),
            CallContext(node="extract_memory", story_id=story_id, ep_no=ep, run_id=run_id, budgeted=False),
            temperature=0.0, max_tokens=3000,
        ).model_dump()
        with transaction(self.conn):
            result = save_memory(self.conn, story_id, ep, vid, v["text"], out, {t["key"]: t for t in threads})
            self.conn.execute("UPDATE episode_versions SET status = 'in_review' WHERE id = ?", (vid,))
        log_step(self.conn, "extract_memory", "extracted", run_id=run_id, story_id=story_id, ep_no=ep,
                 detail={"version_id": vid, **result})
        return result

    def _forget_version(self, vid: int) -> None:
        for table in ("character_states", "relationships", "facts", "thread_events", "key_lines"):
            self.conn.execute(f"DELETE FROM {table} WHERE source_version_id = ?", (vid,))
        self.conn.execute("DELETE FROM threads WHERE source_version_id = ? AND id NOT IN (SELECT thread_id FROM thread_events)", (vid,))


    def needs_review(self, story_id: int, vid: int) -> tuple[bool, str]:
        v = self._version(vid)
        ep = v["ep_no"]
        report = json.loads(v["check_report"] or '{"problems": []}')
        if any(p["severity"] == "must_fix" for p in report["problems"]):
            return True, "problems the checks could not fix"
        r = current_review_setting(self.conn, story_id)
        if r["mode"] == "every_episode":
            return True, "every episode is reviewed"
        if r["mode"] == "every_n" and ep % r["every_n"] == 0:
            return True, f"every {r['every_n']} episodes are reviewed"
        if r["mode"] == "arc_end" and any(a["end_ep"] == ep for a in load_arcs(self.conn, v["plan_version_id"])):
            return True, "end of an arc"
        return False, f"review setting is {r['mode']}"

    def approve(self, story_id: int, vid: int, note: str | None, by_human: bool, run_id: int | None) -> None:
        v = self._version(vid)
        with transaction(self.conn):
            approve_version(self.conn, story_id, vid)
            if by_human:
                self._feedback(story_id, v, "approve", note, None)
        log_step(self.conn, "approve", "approved_by_human" if by_human else "approved_automatically",
                 run_id=run_id, story_id=story_id, ep_no=v["ep_no"], detail={"version_id": vid})

    def human_edit(self, story_id: int, vid: int, text: str, note: str | None) -> int:
        v = self._version(vid)
        text = text.strip()
        if not text:
            raise EpisodeDecisionError("the edited episode is empty")
        with transaction(self.conn):
            self.conn.execute("UPDATE episode_versions SET status = 'superseded' WHERE id = ?", (vid,))
            new = self._new_version(story_id, v["ep_no"], v["plan_version_id"], text, "human",
                                    context_id=v["context_id"], parent_id=vid, revisions=v["revisions"])
            self.conn.execute("UPDATE episode_versions SET check_report = ? WHERE id = ?",
                              (to_json({"problems": [], "dropped_unverified": [], "note": "edited by the reviewer"}), new))
            self._feedback(story_id, v, "edit", note, {"new_version_id": new})
        return new

    def reject(self, story_id: int, vid: int, reason: str, classification: dict[str, Any] | None = None) -> None:
        v = self._version(vid)
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE episode_versions SET status = 'rejected', decided_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (vid,),
            )
            self._feedback(story_id, v, "reject", reason, classification)


    def summarize_finished_arcs(self, story_id: int, pv: int, run_id: int | None) -> list[int]:
        """Summarise every finished arc that has no current summary (new, or made stale by a change).
        The model writes what happened; code fills in everything else from the tables."""
        made = []
        for arc in arcs_to_summarize(self.conn, story_id, pv):
            episodes = [e for e in approved_episodes(self.conn, story_id, arc["end_ep"] + 1) if e["ep_no"] >= arc["start_ep"]]
            narrative = self.llm.complete(
                E.arc_summary_prompt(arc, episodes),
                CallContext(node="summarize_arc", story_id=story_id, ep_no=arc["end_ep"], run_id=run_id, budgeted=False),
                temperature=0.2, max_tokens=500,
            ).text.strip()
            with transaction(self.conn):
                sid = save_summary(self.conn, story_id, arc["start_ep"], arc["end_ep"], narrative,
                                   arc_snapshot(self.conn, story_id, arc["start_ep"], arc["end_ep"]))
            log_step(self.conn, "summarize_arc", "summarized", run_id=run_id, story_id=story_id, ep_no=arc["end_ep"],
                     detail={"arc": arc["arc_no"], "summary_id": sid, "words": len(narrative.split())})
            made.append(sid)
        return made


    def sort_feedback(self, story_id: int, vid: int, text: str, run_id: int | None) -> dict[str, Any]:
        """The model's guess at what kind of feedback this is. The human confirms it."""
        v = self._version(vid)
        beat = load_beats(self.conn, v["plan_version_id"], v["ep_no"], v["ep_no"])[0]
        try:
            out = self.llm.structured(E.feedback_prompt(text, v["ep_no"], beat["beat"]), FeedbackSort,
                                      CallContext(node="sort_feedback", story_id=story_id, ep_no=v["ep_no"],
                                                  run_id=run_id, budgeted=False),
                                      temperature=0.0, max_tokens=500).model_dump()
        except LLMError as exc:
            out = {"kind": "lasting_instruction", "instruction": text, "instruction_kind": "other",
                   "why": f"could not sort automatically ({exc})"}
        with transaction(self.conn):
            fid = self._feedback(story_id, v, "note", text, {"suggested": out})
        return {**out, "feedback_id": fid, "text": text}

    def record_feedback(self, story_id: int, vid: int, text: str, kind: str) -> dict[str, Any]:
        """Feedback the human has already said how to use: no sorting, no confirm step."""
        v = self._version(vid)
        out = {"kind": kind, "instruction": text, "instruction_kind": "other", "why": "chosen by the reviewer"}
        with transaction(self.conn):
            fid = self._feedback(story_id, v, "note", text, {"chosen": kind})
        return {**out, "feedback_id": fid, "text": text}

    def apply_feedback(
        self, story_id: int, vid: int, pv: int, sorted_fb: dict[str, Any], decision: dict[str, Any], run_id: int | None,
    ) -> dict[str, Any]:
        """Apply feedback as the human confirmed it. Returns what the graph does next:
        {"rewrite": note or None, "plan_version_id": pv}."""
        kind = decision.get("kind")
        instruction = (decision.get("instruction") or sorted_fb["instruction"]).strip()
        if kind not in ("fix_episode", "lasting_instruction", "story_change"):
            raise EpisodeDecisionError("pick what kind of feedback this is")
        if not instruction:
            raise EpisodeDecisionError("the instruction is empty")
        v = self._version(vid)
        ep = v["ep_no"]
        confirmed = {"kind": kind, "instruction": instruction, "suggested": sorted_fb["kind"]}

        if kind == "fix_episode":
            self.reject(story_id, vid, instruction, {"feedback_id": sorted_fb["feedback_id"], **confirmed})
            log_step(self.conn, "feedback", "fix_episode", run_id=run_id, story_id=story_id, ep_no=ep, detail=confirmed)
            return {"rewrite": instruction, "plan_version_id": pv}

        if kind == "lasting_instruction":
            with transaction(self.conn):
                did = self._directive(story_id, instruction, decision.get("instruction_kind") or sorted_fb["instruction_kind"],
                                      sorted_fb["feedback_id"], ep)
            log_step(self.conn, "feedback", "lasting_instruction", run_id=run_id, story_id=story_id, ep_no=ep,
                     detail={**confirmed, "directive_id": did})
            if decision.get("rewrite"):
                self.reject(story_id, vid, instruction, {"feedback_id": sorted_fb["feedback_id"], **confirmed})
                return {"rewrite": instruction, "plan_version_id": pv}
            return {"rewrite": None, "plan_version_id": pv}

        new_pv, changed = self.replan_rest_of_arc(story_id, pv, ep, instruction, run_id)
        self.reject(story_id, vid, instruction, {"feedback_id": sorted_fb["feedback_id"], **confirmed,
                                                 "new_plan_version_id": new_pv})
        log_step(self.conn, "feedback", "story_change", run_id=run_id, story_id=story_id, ep_no=ep,
                 detail={**confirmed, "plan_version_id": new_pv, "replanned_eps": changed})
        return {"rewrite": instruction, "plan_version_id": new_pv}

    def replan_rest_of_arc(self, story_id: int, pv: int, ep: int, change: str, run_id: int | None) -> tuple[int, list[int]]:
        """New plan lines for episodes ep..end of arc, as a new approved plan version."""
        arc = next(a for a in load_arcs(self.conn, pv) if a["start_ep"] <= ep <= a["end_ep"])
        old = load_beats(self.conn, pv, ep, arc["end_ep"])
        plan = get_plan_version(self.conn, pv)
        bible = load_bible(self.conn, plan["bible_version_id"])
        names = [c["name"] for c in self._cast(story_id, pv)]
        keys = [t["key"] for t in load_threads(self.conn, pv)]
        written = [e for e in approved_episodes(self.conn, story_id, ep) if e["ep_no"] >= arc["start_ep"]]
        out = self.llm.structured(
            E.replan_prompt(render_bible(bible), arc, written, old, change), beats_model(len(old), names, keys),
            CallContext(node="replan_arc", story_id=story_id, ep_no=ep, run_id=run_id, budgeted=False),
            temperature=0.7, max_tokens=4000,
        )
        with transaction(self.conn):
            new_pv = new_plan_version(self.conn, story_id, plan["bible_version_id"], created_by="human",
                                      reason=f"story change at ep {ep}: {change}", parent_id=pv)
            copy_plan(self.conn, pv, new_pv)
            self.conn.execute("DELETE FROM plan_beats WHERE plan_version_id = ? AND ep_no BETWEEN ? AND ?",
                              (new_pv, ep, arc["end_ep"]))
            for b, beat in zip(old, out.beats):
                self.conn.execute(
                    """INSERT INTO plan_beats (plan_version_id, ep_no, beat, hook, characters, threads, is_turning_point)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (new_pv, b["ep_no"], beat.beat, beat.hook, to_json(list(dict.fromkeys(beat.characters))),
                     to_json([m.model_dump() for m in beat.threads]), b["is_turning_point"]),
                )
            self.conn.execute("UPDATE plan_versions SET status = 'superseded' WHERE story_id = ? AND status = 'approved'",
                              (story_id,))
            self.conn.execute("UPDATE plan_versions SET status = 'approved' WHERE id = ?", (new_pv,))
        return new_pv, [b["ep_no"] for b in old]

    def _directive(self, story_id: int, text: str, kind: str, feedback_id: int | None, ep: int) -> int:
        did = self.conn.execute(
            "INSERT INTO directives (story_id, text, kind, from_feedback_id, from_ep) VALUES (?, ?, ?, ?, ?)",
            (story_id, text, kind, feedback_id, ep),
        ).lastrowid
        self.conn.execute("INSERT INTO directive_status (directive_id, status, ep_no) VALUES (?, 'active', ?)", (did, ep))
        return did

    def _feedback(self, story_id: int, v: sqlite3.Row, action: str, text: str | None, detail: Any) -> int:
        return self.conn.execute(
            """INSERT INTO feedback (story_id, ep_no, episode_version_id, plan_version_id, action, text, classification)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (story_id, v["ep_no"], v["id"], v["plan_version_id"], action, text,
             to_json(detail) if detail is not None else None),
        ).lastrowid


    def _version(self, vid: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM episode_versions WHERE id = ?", (vid,)).fetchone()

    def _pack(self, v: sqlite3.Row) -> str:
        return self.conn.execute("SELECT pack FROM episode_contexts WHERE id = ?", (v["context_id"],)).fetchone()["pack"]

    def _cast(self, story_id: int, pv: int) -> list[dict[str, Any]]:
        """The plan's cast plus people the story itself has introduced."""
        cast = load_cast(self.conn, pv)
        seen = {c["name"] for c in cast}
        extra = [dict(r) for r in self.conn.execute(
            "SELECT * FROM live_characters WHERE story_id = ? AND source_version_id IS NOT NULL ORDER BY id", (story_id,)
        ) if r["name"] not in seen]
        return cast + extra

    @staticmethod
    def _ctx(node: str, story_id: int, ep: int, run_id: int | None, since: str) -> CallContext:
        return CallContext(node=node, story_id=story_id, ep_no=ep, run_id=run_id, budget_since=since)


def _problem(kind: str, message: str, severity: str, source: str = "model", quote: str | None = None) -> dict[str, Any]:
    return {"kind": kind, "message": message, "severity": severity, "source": source, "quote": quote}
