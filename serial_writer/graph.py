"""The story's LangGraph: planning, then (later steps) the episode loop.

Human stops are `interrupt()` calls. The graph pauses, its place is saved by the
checkpointer, and the web app resumes it with the human's decision.

Two rules keep resume safe:
- the graph state holds only ids and flags; the story itself lives in story.db;
- a node that interrupts does nothing else. LangGraph re-runs an interrupted node
  from the top when it resumes, so any work before the interrupt would happen twice.
"""

from __future__ import annotations

import sqlite3
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .llm import LLMClient
from .plan_check import run_plan_check
from .plan_review import PlanDecisionError, apply_plan_decision
from .planner import Planner
from .story import get_story, set_status
from .tracing import log_step


class StoryState(TypedDict, total=False):
    story_id: int
    plan_version_id: int | None
    # Set while part of the plan is being rebuilt: {"target", "no", "note"}.
    redo: dict[str, Any] | None
    decision: dict[str, Any] | None
    review_error: str | None
    outcome: str | None


def _run_id(config: RunnableConfig) -> int | None:
    return (config.get("configurable") or {}).get("run_id")


def _note(state: StoryState) -> str | None:
    return (state.get("redo") or {}).get("note")


def build_graph(conn: sqlite3.Connection, llm: LLMClient, checkpointer: Any) -> Any:
    planner = Planner(conn, llm)

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
        return {
            "plan_version_id": result["plan_version_id"],
            "redo": result["redo"],
            "decision": None,
            "outcome": result["outcome"],
        }

    def after_apply(state: StoryState) -> str:
        if state.get("review_error"):
            return "review_plan"
        if state["outcome"] == "approved":
            return END
        if state["outcome"] == "edited":
            return "check_plan"
        return {"bible": "setup", "all": "plan_acts", "act": "plan_arcs", "arc": "plan_beats"}[state["redo"]["target"]]

    g = StateGraph(StoryState)
    for name, fn in [
        ("setup", setup), ("plan_acts", plan_acts), ("plan_arcs", plan_arcs), ("plan_beats", plan_beats),
        ("check_plan", check_plan), ("review_plan", review_plan), ("apply_plan", apply_plan),
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
        ["review_plan", "check_plan", "setup", "plan_acts", "plan_arcs", "plan_beats", END],
    )
    return g.compile(checkpointer=checkpointer)
