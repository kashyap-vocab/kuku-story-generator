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


_NOTHING_TO_DO = object()


class Engine:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.conn = connect(settings.db_path)
        init_db(self.conn)
        ckpt_path = settings.db_path.with_name("checkpoints.db")
        self._ckpt_conn = sqlite3.connect(str(ckpt_path), check_same_thread=False)
        self.llm = LLMClient(settings, self.conn, client=client)
        self._stop: set[int] = set()
        self.graph = build_graph(self.conn, self.llm, SqliteSaver(self._ckpt_conn),
                                 max_revisions=settings.max_revisions, stop_requested=lambda sid: sid in self._stop)
        self._running: set[int] = set()
        self._guard = threading.Lock()
        self.errors: dict[int, str] = {}

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
        payload = self._claim(story_id, decision)
        if payload is _NOTHING_TO_DO:
            return self.status(story_id)
        return self._run(story_id, payload)

    def start(self, story_id: int, decision: dict[str, Any] | None = None) -> bool:
        """Like `advance`, but runs in a background thread so the web app stays quick.
        Bad requests (wrong state, already running) raise here, before the thread starts.
        Returns False if there was nothing to run."""
        payload = self._claim(story_id, decision)
        if payload is _NOTHING_TO_DO:
            return False
        self.errors.pop(story_id, None)

        def work() -> None:
            try:
                self._run(story_id, payload)
            except BaseException as exc:
                self.errors[story_id] = f"{type(exc).__name__}: {exc}"

        threading.Thread(target=work, name=f"story-{story_id}", daemon=True).start()
        return True

    def request_stop(self, story_id: int) -> None:
        """Pause after the episode being written: the story then waits to be told how far to write."""
        self._stop.add(story_id)

    def stop_requested(self, story_id: int) -> bool:
        return story_id in self._stop

    def _claim(self, story_id: int, decision: dict[str, Any] | None) -> Any:
        """Check the story can move on, mark it running, and return what to send the graph."""
        with self._guard:
            current = self.status(story_id)
            if current["state"] == "running":
                raise RuntimeError(f"story {story_id} is already running")
            if current["state"] != "waiting" and decision is not None:
                raise RuntimeError(f"story {story_id} is not waiting for a decision ({current['state']})")
            if current["state"] == "done" or (current["state"] == "waiting" and decision is None):
                return _NOTHING_TO_DO
            self._running.add(story_id)
            self._stop.discard(story_id)

        if decision is not None:
            return Command(resume=decision)
        if current["state"] == "not_started":
            return {"story_id": story_id}
        return None

    def _run(self, story_id: int, payload: Any) -> dict[str, Any]:
        try:
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
