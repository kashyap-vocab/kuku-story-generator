"""Runs a story's graph: start it, resume it after a human decision, or pick up
after a crash. The web app and CLI both drive stories through this."""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import Settings
from .db import connect, init_db
from .graph import build_graph
from .llm import LLMClient
from .story import create_story
from .tracing import finish_run, start_run


class Engine:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.conn = connect(settings.db_path)
        init_db(self.conn)
        # Graph checkpoints live in their own file, so they never contend with story reads.
        ckpt_path = settings.db_path.with_name("checkpoints.db")
        self._ckpt_conn = sqlite3.connect(str(ckpt_path), check_same_thread=False)
        self.llm = LLMClient(settings, self.conn, client=client)
        self.graph = build_graph(self.conn, self.llm, SqliteSaver(self._ckpt_conn))
        self._running: set[int] = set()
        self._guard = threading.Lock()

    def close(self) -> None:
        self._ckpt_conn.close()
        self.conn.close()

    def create_story(self, premise: str, **kwargs: Any) -> int:
        return create_story(self.conn, premise, **kwargs)

    @staticmethod
    def _config(story_id: int, run_id: int | None = None) -> dict[str, Any]:
        return {"configurable": {"thread_id": f"story-{story_id}", "run_id": run_id}}

    def status(self, story_id: int) -> dict[str, Any]:
        """not_started | running | waiting (for a human) | paused (stopped mid-way) | done."""
        if story_id in self._running:
            return {"state": "running"}
        snap = self.graph.get_state(self._config(story_id))
        if not snap.values:
            return {"state": "not_started"}
        if snap.interrupts:
            return {"state": "waiting", "waiting_for": snap.interrupts[0].value}
        if snap.next:
            return {"state": "paused", "next": list(snap.next)}
        return {"state": "done"}

    def advance(self, story_id: int, decision: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the story until it needs a human or finishes.

        With `decision`: answer the human stop it is waiting at.
        Without: start it, or continue it after a crash or restart.
        """
        current = self.status(story_id)
        if current["state"] == "running":
            raise RuntimeError(f"story {story_id} is already running")
        if current["state"] == "waiting" and decision is None:
            return current
        if current["state"] != "waiting" and decision is not None:
            raise RuntimeError(f"story {story_id} is not waiting for a decision ({current['state']})")
        if current["state"] == "done":
            return current

        if decision is not None:
            payload: Any = Command(resume=decision)
        elif current["state"] == "not_started":
            payload = {"story_id": story_id}
        else:
            payload = None  # carry on from the last checkpoint

        with self._guard:
            if story_id in self._running:
                raise RuntimeError(f"story {story_id} is already running")
            self._running.add(story_id)
        run_id = start_run(self.conn, "graph", story_id)
        try:
            self.graph.invoke(payload, self._config(story_id, run_id))
        except BaseException as exc:
            finish_run(self.conn, run_id, "error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with self._guard:
                self._running.discard(story_id)
        result = self.status(story_id)
        finish_run(self.conn, run_id, "ok", result["state"])
        return result
