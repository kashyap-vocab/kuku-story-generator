"""The story's LangGraph: planning, then the episode loop.

Human stops are `interrupt()` calls. The graph pauses, its place is saved by the
checkpointer, and the web app resumes it with the human's decision.

Two rules keep resume safe:
- the graph state holds only ids and flags; the story itself lives in story.db;
- a node that interrupts does nothing else. LangGraph re-runs an interrupted node
  from the top when it resumes, so any work before the interrupt would happen twice.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .llm import LLMClient
from .memory import next_episode_no
from .plan_check import run_plan_check
from .plan_review import PlanDecisionError, apply_plan_decision
from .planner import Planner
from .story import get_story, set_status
from .tracing import log_step
from .writer import EpisodeDecisionError, Writer, now


class StoryState(TypedDict, total=False):
    story_id: int
    plan_version_id: int | None
    redo: dict[str, Any] | None
    decision: dict[str, Any] | None
    review_error: str | None
    outcome: str | None
    write_until: int
    ep_no: int | None
    version_id: int | None
    revisions: int
    attempt_started: str | None
    rewrite_note: str | None
    human_edited: bool
    feedback: dict[str, Any] | None


def _run_id(config: RunnableConfig) -> int | None:
    return (config.get("configurable") or {}).get("run_id")


def _note(state: StoryState) -> str | None:
    return (state.get("redo") or {}).get("note")


def build_graph(
    conn: sqlite3.Connection, llm: LLMClient, checkpointer: Any, max_revisions: int = 2,
    stop_requested: Callable[[int], bool] = lambda story_id: False,
) -> Any:
    planner = Planner(conn, llm)
    writer = Writer(conn, llm)

    def setup(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        pv = planner.setup(state["story_id"], state.get("plan_version_id"), state.get("redo"), _run_id(config))
        set_status(conn, state["story_id"], "planning")
        return {"plan_version_id": pv}

    def plan_acts(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        planner.fill_acts(state["story_id"], state["plan_version_id"], _note(state), _run_id(config))
        return {}

    def plan_arcs(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        planner.fill_arcs(state["story_id"], state["plan_version_id"], _note(state), _run_id(config))
        return {}

    def plan_beats(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        planner.fill_beats(state["story_id"], state["plan_version_id"], _note(state), _run_id(config))
        return {}

    def check_plan(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        story = get_story(conn, state["story_id"])
        run_plan_check(conn, llm, state["story_id"], state["plan_version_id"], story["total_episodes"], _run_id(config))
        set_status(conn, state["story_id"], "plan_review")
        return {"redo": None}

    def review_plan(state: StoryState) -> dict[str, Any]:
        decision = interrupt({
            "kind": "plan_review",
            "story_id": state["story_id"],
            "plan_version_id": state["plan_version_id"],
            "error": state.get("review_error"),
        })
        return {"decision": decision, "review_error": None}

    def apply_plan(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        try:
            result = apply_plan_decision(conn, state["story_id"], state["plan_version_id"], state["decision"] or {})
        except PlanDecisionError as exc:
            log_step(conn, "apply_plan", "decision_rejected", run_id=_run_id(config),
                     story_id=state["story_id"], detail={"error": str(exc)})
            return {"review_error": str(exc), "decision": None, "outcome": None}
        log_step(conn, "apply_plan", result["outcome"], run_id=_run_id(config), story_id=state["story_id"],
                 detail={"plan_version_id": result["plan_version_id"], "redo": result["redo"]})
        out = {
            "plan_version_id": result["plan_version_id"],
            "redo": result["redo"],
            "decision": None,
            "outcome": result["outcome"],
        }
        if result["outcome"] == "approved":
            out["write_until"] = _int(state["decision"].get("write_until"))
        return out

    def after_apply(state: StoryState) -> str:
        if state.get("review_error"):
            return "review_plan"
        if state["outcome"] == "approved":
            return "next_episode"
        if state["outcome"] == "edited":
            return "check_plan"
        return {"bible": "setup", "all": "plan_acts", "act": "plan_arcs", "arc": "plan_beats"}[state["redo"]["target"]]


    def next_episode(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        sid = state["story_id"]
        writer.summarize_finished_arcs(sid, state["plan_version_id"], _run_id(config))
        ep = next_episode_no(conn, sid)
        if ep > get_story(conn, sid)["total_episodes"]:
            set_status(conn, sid, "done")
            return {"ep_no": None}
        set_status(conn, sid, "writing")
        return {"ep_no": ep, "version_id": None, "revisions": 0, "rewrite_note": None,
                "human_edited": False, "feedback": None}

    def after_next(state: StoryState) -> str:
        if state.get("ep_no") is None:
            return END
        if stop_requested(state["story_id"]) or state["ep_no"] > state.get("write_until", 0):
            return "wait_to_write"
        return "draft_episode"

    def wait_to_write(state: StoryState) -> dict[str, Any]:
        decision = interrupt({"kind": "write_more", "story_id": state["story_id"], "next_ep": state["ep_no"],
                              "error": state.get("review_error")})
        return {"decision": decision, "review_error": None}

    def apply_write_more(state: StoryState) -> dict[str, Any]:
        until = _int((state.get("decision") or {}).get("until"))
        if until < state["ep_no"]:
            return {"decision": None, "review_error": f"write up to an episode from {state['ep_no']} on"}
        set_status(conn, state["story_id"], "writing")
        return {"decision": None, "write_until": until}

    def draft_episode(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        started = now()
        vid = writer.draft(state["story_id"], state["ep_no"], state["plan_version_id"],
                           state.get("rewrite_note"), started, _run_id(config))
        return {"version_id": vid, "revisions": 0, "attempt_started": started, "human_edited": False}

    def check_episode(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        writer.check(state["story_id"], state["version_id"], state["attempt_started"], _run_id(config))
        return {}

    def after_check(state: StoryState) -> str:
        report = conn.execute("SELECT check_report FROM episode_versions WHERE id = ?",
                              (state["version_id"],)).fetchone()[0]
        must = [p for p in json.loads(report)["problems"] if p["severity"] == "must_fix"]
        if must and state["revisions"] < max_revisions and not any(p["kind"] == "budget" for p in must):
            return "revise_episode"
        return "extract_memory"

    def revise_episode(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        new = writer.revise(state["story_id"], state["version_id"], state["attempt_started"], _run_id(config))
        if new is None:
            return {"revisions": max_revisions}
        return {"version_id": new, "revisions": state["revisions"] + 1}

    def after_revise(state: StoryState) -> str:
        return "check_episode" if _is_new(conn, state) else "extract_memory"

    def extract_memory(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        writer.extract(state["story_id"], state["version_id"], _run_id(config))
        return {}

    def route_review(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        if state.get("human_edited"):
            return {"outcome": "approve"}
        needed, why = writer.needs_review(state["story_id"], state["version_id"])
        log_step(conn, "route_review", "to_human" if needed else "auto_approve", run_id=_run_id(config),
                 story_id=state["story_id"], ep_no=state["ep_no"], detail={"why": why})
        return {"outcome": "review" if needed else "approve"}

    def approve_episode(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        writer.approve(state["story_id"], state["version_id"], None, by_human=False, run_id=_run_id(config))
        return {}

    def review_episode(state: StoryState) -> dict[str, Any]:
        decision = interrupt({"kind": "episode_review", "story_id": state["story_id"], "ep_no": state["ep_no"],
                              "version_id": state["version_id"], "error": state.get("review_error")})
        return {"decision": decision, "review_error": None}

    def apply_episode(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        d = state.get("decision") or {}
        sid, vid = state["story_id"], state["version_id"]
        action = d.get("action")
        try:
            if action == "approve":
                writer.approve(sid, vid, d.get("note"), by_human=True, run_id=_run_id(config))
                return {"decision": None, "outcome": "approved"}
            if action == "edit":
                new = writer.human_edit(sid, vid, d.get("text") or "", d.get("note"))
                return {"decision": None, "outcome": "edited", "version_id": new, "human_edited": True}
            if action == "reject":
                reason = (d.get("reason") or "").strip()
                if not reason:
                    raise EpisodeDecisionError("say why, so the rewrite can fix it")
                writer.reject(sid, vid, reason)
                return {"decision": None, "outcome": "rewrite", "rewrite_note": reason}
            if action == "feedback":
                text = (d.get("text") or "").strip()
                if not text:
                    raise EpisodeDecisionError("the feedback is empty")
                kind = d.get("kind")
                if kind in ("fix_episode", "lasting_instruction", "story_change"):
                    return {"decision": {"kind": kind, "instruction": text, "rewrite": True},
                            "outcome": "feedback_chosen", "feedback": writer.record_feedback(sid, vid, text, kind)}
                return {"decision": None, "outcome": "feedback",
                        "feedback": writer.sort_feedback(sid, vid, text, _run_id(config))}
            raise EpisodeDecisionError(f"unknown action {action!r}")
        except EpisodeDecisionError as exc:
            return {"decision": None, "outcome": None, "review_error": str(exc)}

    def after_episode_decision(state: StoryState) -> str:
        return {"approved": "next_episode", "edited": "extract_memory", "rewrite": "draft_episode",
                "feedback": "confirm_feedback", "feedback_chosen": "apply_feedback"}.get(
                    state.get("outcome") or "", "review_episode")

    def confirm_feedback(state: StoryState) -> dict[str, Any]:
        decision = interrupt({"kind": "feedback_confirm", "story_id": state["story_id"], "ep_no": state["ep_no"],
                              "version_id": state["version_id"], "feedback": state["feedback"],
                              "error": state.get("review_error")})
        return {"decision": decision, "review_error": None}

    def apply_feedback(state: StoryState, config: RunnableConfig) -> dict[str, Any]:
        d = state.get("decision") or {}
        if d.get("kind") == "cancel":
            return {"decision": None, "feedback": None, "outcome": "back"}
        try:
            result = writer.apply_feedback(state["story_id"], state["version_id"], state["plan_version_id"],
                                           state["feedback"], d, _run_id(config))
        except EpisodeDecisionError as exc:
            return {"decision": None, "outcome": None, "review_error": str(exc)}
        return {"decision": None, "feedback": None, "plan_version_id": result["plan_version_id"],
                "rewrite_note": result["rewrite"], "outcome": "rewrite" if result["rewrite"] else "back"}

    def after_feedback(state: StoryState) -> str:
        return {"rewrite": "draft_episode", "back": "review_episode"}.get(state.get("outcome") or "", "confirm_feedback")

    g = StateGraph(StoryState)
    for name, fn in [
        ("setup", setup), ("plan_acts", plan_acts), ("plan_arcs", plan_arcs), ("plan_beats", plan_beats),
        ("check_plan", check_plan), ("review_plan", review_plan), ("apply_plan", apply_plan),
        ("next_episode", next_episode), ("wait_to_write", wait_to_write), ("apply_write_more", apply_write_more),
        ("draft_episode", draft_episode), ("check_episode", check_episode), ("revise_episode", revise_episode),
        ("extract_memory", extract_memory), ("route_review", route_review), ("approve_episode", approve_episode),
        ("review_episode", review_episode), ("apply_episode", apply_episode),
        ("confirm_feedback", confirm_feedback), ("apply_feedback", apply_feedback),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "setup")
    g.add_edge("setup", "plan_acts")
    g.add_edge("plan_acts", "plan_arcs")
    g.add_edge("plan_arcs", "plan_beats")
    g.add_edge("plan_beats", "check_plan")
    g.add_edge("check_plan", "review_plan")
    g.add_edge("review_plan", "apply_plan")
    g.add_conditional_edges(
        "apply_plan", after_apply,
        ["review_plan", "check_plan", "setup", "plan_acts", "plan_arcs", "plan_beats", "next_episode"],
    )

    g.add_conditional_edges("next_episode", after_next, ["wait_to_write", "draft_episode", END])
    g.add_edge("wait_to_write", "apply_write_more")
    g.add_conditional_edges("apply_write_more", lambda s: "wait_to_write" if s.get("review_error") else "next_episode",
                            ["wait_to_write", "next_episode"])
    g.add_edge("draft_episode", "check_episode")
    g.add_conditional_edges("check_episode", after_check, ["revise_episode", "extract_memory"])
    g.add_conditional_edges("revise_episode", after_revise, ["check_episode", "extract_memory"])
    g.add_edge("extract_memory", "route_review")
    g.add_conditional_edges("route_review", lambda s: "review_episode" if s["outcome"] == "review" else "approve_episode",
                            ["review_episode", "approve_episode"])
    g.add_edge("approve_episode", "next_episode")
    g.add_edge("review_episode", "apply_episode")
    g.add_conditional_edges("apply_episode", after_episode_decision,
                            ["next_episode", "extract_memory", "draft_episode", "confirm_feedback", "apply_feedback",
                             "review_episode"])
    g.add_edge("confirm_feedback", "apply_feedback")
    g.add_conditional_edges("apply_feedback", after_feedback, ["draft_episode", "review_episode", "confirm_feedback"])
    return g.compile(checkpointer=checkpointer)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_new(conn: sqlite3.Connection, state: StoryState) -> bool:
    """A revision that hasn't been checked yet (the last allowed one still gets its check)."""
    row = conn.execute("SELECT check_report FROM episode_versions WHERE id = ?", (state["version_id"],)).fetchone()
    return row is not None and row[0] is None
