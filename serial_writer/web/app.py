"""The web app: start a story, review and steer the plan, review each episode,
give feedback that carries forward, and control how far it writes.

Pages are plain HTML forms; HTMX only polls progress while the model works.
The story's graph runs in a background thread (Engine.start), and every
decision goes through the graph, so the web app and the graph never disagree
about where a story is.

Each request reads through its own SQLite connection, so it never lands inside
a transaction the writer has open on the engine's connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from ..db import connect
from ..engine import Engine
from ..memory import character_states, directives, next_episode_no, proposed_memory, relationships, thread_states
from ..plan_models import BeatEdit, PlanChanges
from ..planner import plain_title
from ..plan_store import get_plan_version, load_acts, load_arcs, load_beats, load_bible, load_cast, load_threads
from ..story import REVIEW_MODES, create_story, current_review_setting, get_story, set_review_setting

HERE = Path(__file__).parent


class PendingEdits:
    """Plan edits the reviewer has made but not sent yet, per plan version.

    Sending an edit re-checks the whole plan (a few model calls), so edits are
    collected here and sent together. Kept in memory: an unsent edit is lost if
    the server restarts, which only costs the reviewer retyping it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_plan: dict[int, dict[str, Any]] = {}

    def get(self, pv: int) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._by_plan.get(pv) or _empty_changes()))

    def put(self, pv: int, section: str, key: Any, fields: dict[str, Any]) -> None:
        with self._lock:
            changes = self._by_plan.setdefault(pv, _empty_changes())
            if section == "cast":
                changes["bible"].setdefault("cast", {})[key] = fields
            elif section == "bible":
                changes["bible"].update(fields)
            else:
                changes[section][str(key)] = fields

    def count(self, pv: int) -> int:
        c = self.get(pv)
        bible = {k: v for k, v in c["bible"].items() if k != "cast"}
        return len(bible) + len(c["bible"].get("cast", {})) + len(c["acts"]) + len(c["arcs"]) + len(c["beats"])

    def clear(self, pv: int) -> None:
        with self._lock:
            self._by_plan.pop(pv, None)


def _empty_changes() -> dict[str, Any]:
    return {"bible": {}, "acts": {}, "arcs": {}, "beats": {}}


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Serial story writer")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["plain_title"] = plain_title
    templates.env.globals["css_v"] = int((HERE / "static" / "style.css").stat().st_mtime)
    pending = PendingEdits()

    def db() -> Iterator[sqlite3.Connection]:
        conn = connect(engine.settings.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    def back(url: str, error: str | None = None) -> RedirectResponse:
        if error:
            url += ("&" if "?" in url else "?") + "error=" + quote(error)
        return RedirectResponse(url, status_code=303)

    def story_or_404(conn: sqlite3.Connection, sid: int) -> sqlite3.Row:
        try:
            return get_story(conn, sid)
        except KeyError:
            raise HTTPException(404, f"no story {sid}")

    def status_of(sid: int) -> dict[str, Any]:
        st = engine.status(sid)
        st["error"] = engine.errors.get(sid)
        return st

    def decide(sid: int, decision: dict[str, Any], to: str) -> RedirectResponse:
        try:
            engine.start(sid, decision)
        except RuntimeError as exc:
            return back(to, str(exc))
        return back(f"/stories/{sid}")


    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, conn: sqlite3.Connection = Depends(db)):
        stories = conn.execute("SELECT * FROM stories ORDER BY id DESC").fetchall()
        return page(request, "index.html", stories=stories, modes=REVIEW_MODES,
                    states={s["id"]: engine.status(s["id"])["state"] for s in stories})

    @app.post("/stories")
    def new_story(
        premise: str = Form(...), episodes: int = Form(200), review_mode: str = Form("every_episode"),
        every_n: int | None = Form(None), conn: sqlite3.Connection = Depends(db),
    ):
        try:
            sid = create_story(conn, premise, total_episodes=episodes, review_mode=review_mode,
                               every_n=every_n if review_mode == "every_n" else None)
        except ValueError as exc:
            return back("/", str(exc))
        engine.start(sid)
        return back(f"/stories/{sid}")

    @app.get("/stories/{sid}", response_class=HTMLResponse)
    def story_page(request: Request, sid: int, conn: sqlite3.Connection = Depends(db)):
        story = story_or_404(conn, sid)
        return page(request, "story.html", story=story, status=status_of(sid),
                    review=current_review_setting(conn, sid), **_progress(conn, sid))

    @app.get("/stories/{sid}/status", response_class=HTMLResponse)
    def status_fragment(request: Request, sid: int, was: str = "", conn: sqlite3.Connection = Depends(db)):
        status = status_of(sid)
        resp = page(request, "_status.html", story=story_or_404(conn, sid), status=status, **_progress(conn, sid))
        if was and status["state"] != was:
            w = status.get("waiting_for") or {}
            target = {
                "plan_review": f"/stories/{sid}/plan",
                "episode_review": f"/stories/{sid}/episodes/{w.get('ep_no')}",
                "feedback_confirm": f"/stories/{sid}/episodes/{w.get('ep_no')}#confirm",
            }.get(w.get("kind", "")) if status["state"] == "waiting" and not w.get("error") else None
            if target:
                resp.headers["HX-Redirect"] = target
            else:
                resp.headers["HX-Refresh"] = "true"
        return resp

    @app.get("/stories/{sid}/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request, sid: int, conn: sqlite3.Connection = Depends(db)):
        story_or_404(conn, sid)
        return page(request, "_dashboard.html", story_id=sid, budget=engine.settings.episode_token_budget,
                    **_dashboard(conn, sid))

    @app.post("/stories/{sid}/continue")
    def continue_story(sid: int):
        try:
            engine.start(sid)
        except RuntimeError as exc:
            return back(f"/stories/{sid}", str(exc))
        return back(f"/stories/{sid}")


    def plan_version(conn: sqlite3.Connection, sid: int) -> int:
        """The plan waiting for review, else the approved one, else the newest."""
        waiting = engine.status(sid).get("waiting_for") or {}
        if waiting.get("plan_version_id"):
            return waiting["plan_version_id"]
        row = conn.execute(
            "SELECT id FROM plan_versions WHERE story_id = ? ORDER BY (status = 'approved') DESC, version DESC LIMIT 1",
            (sid,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "this story has no plan yet")
        return row["id"]

    @app.get("/stories/{sid}/plan", response_class=HTMLResponse)
    def plan_page(request: Request, sid: int, tab: str = "episodes", conn: sqlite3.Connection = Depends(db)):
        story = story_or_404(conn, sid)
        pv = plan_version(conn, sid)
        plan = get_plan_version(conn, pv)
        status = status_of(sid)
        report = json.loads(plan["check_report"]) if plan["check_report"] else None
        problems = sorted((report or {}).get("problems", []),
                          key=lambda p: (p["severity"] != "must_fix", (p["eps"] or [0])[0]))
        problems_by_ep: dict[int, list[dict[str, Any]]] = {}
        for p in problems:
            for ep in p["eps"]:
                problems_by_ep.setdefault(ep, []).append(p)
        arcs = load_arcs(conn, pv)
        beats = load_beats(conn, pv)
        return page(
            request, "plan.html", story=story, plan=plan, status=status,
            tab=tab if tab in ("episodes", "story", "problems", "history") else "episodes",
            reviewing=(status.get("waiting_for") or {}).get("plan_version_id") == pv,
            bible=load_bible(conn, plan["bible_version_id"]), acts=load_acts(conn, pv),
            arcs_by_act={a["act_no"]: [r for r in arcs if r["act_no"] == a["act_no"]] for a in load_acts(conn, pv)},
            beats_by_arc={a["arc_no"]: [b for b in beats if a["start_ep"] <= b["ep_no"] <= a["end_ep"]] for a in arcs},
            cast=load_cast(conn, pv), threads=load_threads(conn, pv),
            problems=problems, problems_by_ep=problems_by_ep,
            dropped=(report or {}).get("dropped_unverified", 0),
            pending=pending.get(pv), pending_count=pending.count(pv),
            versions=conn.execute(
                "SELECT * FROM plan_versions WHERE story_id = ? ORDER BY version DESC", (sid,)
            ).fetchall(),
        )

    def editable_plan(conn: sqlite3.Connection, sid: int) -> int:
        pv = plan_version(conn, sid)
        if (engine.status(sid).get("waiting_for") or {}).get("plan_version_id") != pv:
            raise HTTPException(409, "the plan is not waiting for review")
        return pv

    @app.post("/stories/{sid}/plan/edit/beat/{ep}")
    def edit_beat(sid: int, ep: int, beat: str = Form(...), hook: str = Form(""), characters: str = Form(""),
                  threads: str = Form(""), conn: sqlite3.Connection = Depends(db)):
        pv = editable_plan(conn, sid)
        to = f"/stories/{sid}/plan?tab=episodes#ep-{ep}"
        moves = []
        for line in threads.splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                return back(f"/stories/{sid}/plan", f"ep {ep}: thread lines look like 'advance the_key', got '{line}'")
            moves.append({"event": parts[0], "key": parts[1]})
        fields = {"beat": beat.strip(), "hook": hook.strip(),
                  "characters": [n.strip() for n in characters.split(",") if n.strip()], "threads": moves}
        try:
            BeatEdit.model_validate(fields)
        except ValidationError as exc:
            return back(f"/stories/{sid}/plan", f"ep {ep}: {_first_error(exc)}")
        pending.put(pv, "beats", ep, fields)
        return back(to)

    @app.post("/stories/{sid}/plan/edit/{section}/{no}")
    def edit_span(sid: int, section: str, no: int, title: str = Form(...), goal: str = Form(...),
                  turning_point: str = Form(""), conn: sqlite3.Connection = Depends(db)):
        if section not in ("acts", "arcs"):
            raise HTTPException(404)
        pv = editable_plan(conn, sid)
        pending.put(pv, section, no, {"title": title.strip(), "goal": goal.strip(),
                                       "turning_point": turning_point.strip()})
        return back(f"/stories/{sid}/plan?tab=episodes#{section[:-1]}-{no}")

    @app.post("/stories/{sid}/plan/edit/bible")
    def edit_bible(sid: int, title: str = Form(...), logline: str = Form(...), tone: str = Form(...),
                   hidden_truth: str = Form(...), style_guide: str = Form(...), world_rules: str = Form(...),
                   conn: sqlite3.Connection = Depends(db)):
        pv = editable_plan(conn, sid)
        pending.put(pv, "bible", None, {
            "title": title.strip(), "logline": logline.strip(), "tone": tone.strip(),
            "hidden_truth": hidden_truth.strip(),
            "style_guide": _lines(style_guide), "world_rules": _lines(world_rules),
        })
        return back(f"/stories/{sid}/plan?tab=story#rules")

    @app.post("/stories/{sid}/plan/edit/cast")
    def edit_cast(sid: int, name: str = Form(...), role: str = Form(...), description: str = Form(...),
                  wants: str = Form(...), secret: str = Form(...), arc: str = Form(...),
                  conn: sqlite3.Connection = Depends(db)):
        pv = editable_plan(conn, sid)
        pending.put(pv, "cast", name, {"role": role.strip(), "description": description.strip(),
                                        "wants": wants.strip(), "secret": secret.strip(), "arc": arc.strip()})
        return back(f"/stories/{sid}/plan?tab=story#cast")

    @app.post("/stories/{sid}/plan/edits/send")
    def send_edits(sid: int, note: str = Form(""), conn: sqlite3.Connection = Depends(db)):
        pv = editable_plan(conn, sid)
        changes = pending.get(pv)
        if not changes["bible"]:
            changes["bible"] = None
        try:
            PlanChanges.model_validate(changes)
        except ValidationError as exc:
            return back(f"/stories/{sid}/plan", _first_error(exc))
        try:
            engine.start(sid, {"action": "edit", "note": note.strip() or None, "changes": changes})
        except RuntimeError as exc:
            return back(f"/stories/{sid}/plan", str(exc))
        pending.clear(pv)
        return back(f"/stories/{sid}")

    @app.post("/stories/{sid}/plan/edits/discard")
    def discard_edits(sid: int, conn: sqlite3.Connection = Depends(db)):
        pending.clear(plan_version(conn, sid))
        return back(f"/stories/{sid}/plan")

    @app.post("/stories/{sid}/plan/redo")
    def redo_plan(sid: int, target: str = Form(...), no: int | None = Form(None), note: str = Form(...),
                  conn: sqlite3.Connection = Depends(db)):
        pv = editable_plan(conn, sid)
        if not note.strip():
            return back(f"/stories/{sid}/plan", "say what should change: the note is what the model rebuilds from")
        pending.clear(pv)
        return decide(sid, {"action": "redo", "target": target, "no": no, "note": note.strip()},
                      f"/stories/{sid}/plan")

    @app.post("/stories/{sid}/plan/approve")
    def approve_plan(sid: int, note: str = Form(""), write_until: int = Form(0),
                     conn: sqlite3.Connection = Depends(db)):
        pv = editable_plan(conn, sid)
        if pending.count(pv):
            return back(f"/stories/{sid}/plan", "you have unsent edits: send or discard them first")
        return decide(sid, {"action": "approve", "note": note.strip() or None, "write_until": write_until},
                      f"/stories/{sid}/plan")


    @app.post("/stories/{sid}/write")
    def write_more(sid: int, until: int = Form(...)):
        return decide(sid, {"until": until}, f"/stories/{sid}")

    @app.post("/stories/{sid}/review")
    def change_review(sid: int, review_mode: str = Form(...), every_n: int | None = Form(None),
                      conn: sqlite3.Connection = Depends(db)):
        story_or_404(conn, sid)
        try:
            set_review_setting(conn, sid, review_mode, every_n if review_mode == "every_n" else None,
                               effective_from_ep=next_episode_no(conn, sid))
        except ValueError as exc:
            return back(f"/stories/{sid}", str(exc))
        return back(f"/stories/{sid}")

    @app.post("/stories/{sid}/pause")
    def pause(sid: int):
        engine.request_stop(sid)
        return back(f"/stories/{sid}")


    @app.get("/stories/{sid}/episodes/{ep}", response_class=HTMLResponse)
    def episode_page(request: Request, sid: int, ep: int, v: int | None = None, edit: int = 0,
                     conn: sqlite3.Connection = Depends(db)):
        story = story_or_404(conn, sid)
        versions = [dict(r) for r in conn.execute(
            "SELECT * FROM episode_versions WHERE story_id = ? AND ep_no = ? ORDER BY version", (sid, ep))]
        status = status_of(sid)
        waiting = status.get("waiting_for") or {}
        if not versions:
            if status["state"] == "running":
                return back(f"/stories/{sid}")
            raise HTTPException(404, f"episode {ep} hasn't been written")
        by_id = {x["id"]: x for x in versions}
        live_id = waiting.get("version_id") if waiting.get("ep_no") == ep else None
        chosen = (by_id.get(v) or by_id.get(live_id)
                  or next((x for x in versions if x["status"] == "approved"), None) or versions[-1])
        report = json.loads(chosen["check_report"]) if chosen["check_report"] else None
        beat = load_beats(conn, chosen["plan_version_id"], ep, ep)
        pack = conn.execute("SELECT pack FROM episode_contexts WHERE id = ?", (chosen["context_id"],)).fetchone()
        cost = conn.execute(
            """SELECT COUNT(*) AS calls, COALESCE(SUM(prompt_tokens), 0) AS tokens_in,
                      COALESCE(SUM(completion_tokens), 0) AS tokens_out, COALESCE(SUM(latency_ms), 0) / 1000 AS seconds
               FROM llm_calls WHERE story_id = ? AND ep_no = ?""", (sid, ep)).fetchone()
        title, _, body = chosen["text"].partition("\n")
        reviewing = waiting.get("kind") == "episode_review" and live_id == chosen["id"]
        return page(
            request, "episode.html", story=story, ep=ep, v=chosen, versions=versions, status=status,
            reviewing=reviewing, editing=reviewing and bool(edit),
            confirming=waiting if waiting.get("kind") == "feedback_confirm" and waiting.get("ep_no") == ep else None,
            title=title.strip(), body=body.strip(),
            problems=(report or {}).get("problems", []), dropped=(report or {}).get("dropped_unverified", []),
            memory=proposed_memory(conn, chosen["id"]), beat=beat[0] if beat else None,
            pack=pack["pack"] if pack else None, cost=cost,
            feedback=conn.execute("SELECT * FROM feedback WHERE story_id = ? AND ep_no = ? ORDER BY id",
                                  (sid, ep)).fetchall(),
            last_ep=conn.execute("SELECT MAX(ep_no) FROM episode_versions WHERE story_id = ?", (sid,)).fetchone()[0],
        )

    @app.post("/stories/{sid}/episodes/{ep}/decide")
    def decide_episode(sid: int, ep: int, action: str = Form(...), note: str = Form(""), text: str = Form(""),
                       reason: str = Form(""), feedback: str = Form(""), kind: str = Form("auto")):
        to = f"/stories/{sid}/episodes/{ep}"
        waiting = engine.status(sid).get("waiting_for") or {}
        if waiting.get("kind") != "episode_review" or waiting.get("ep_no") != ep:
            return back(to, "this episode is not waiting for review")
        decision: dict[str, Any] = {"action": action}
        if action == "approve":
            decision["note"] = note.strip() or None
        elif action == "edit":
            decision.update(text=text.replace("\r\n", "\n"), note=note.strip() or None)
        elif action == "reject":
            decision["reason"] = reason.strip()
        elif action == "feedback":
            decision["text"] = feedback.strip()
            if kind != "auto":
                decision["kind"] = kind
        else:
            raise HTTPException(400, "unknown action")
        return decide(sid, decision, to)

    @app.post("/stories/{sid}/episodes/{ep}/feedback")
    def confirm_feedback(sid: int, ep: int, kind: str = Form(...), instruction: str = Form(""),
                         instruction_kind: str = Form("other"), rewrite: str = Form("")):
        decision = {"kind": kind, "instruction": instruction.strip(), "instruction_kind": instruction_kind,
                    "rewrite": rewrite == "yes"}
        return decide(sid, decision, f"/stories/{sid}/episodes/{ep}")


    @app.get("/stories/{sid}/memory", response_class=HTMLResponse)
    def memory_page(request: Request, sid: int, conn: sqlite3.Connection = Depends(db)):
        story = story_or_404(conn, sid)
        upto = conn.execute(
            "SELECT COALESCE(MAX(ep_no), 0) FROM episode_versions WHERE story_id = ? AND status = 'approved'", (sid,)
        ).fetchone()[0]
        history = {}
        for d in directives(conn, sid, status=None):
            history[d["id"]] = conn.execute(
                "SELECT * FROM directive_status WHERE directive_id = ? ORDER BY id", (d["id"],)).fetchall()
        return page(
            request, "memory.html", story=story, upto=upto,
            states=character_states(conn, sid, upto + 1), rels=relationships(conn, sid, upto + 1),
            threads=thread_states(conn, sid, upto + 1),
            facts=conn.execute("SELECT * FROM live_facts WHERE story_id = ? ORDER BY ep_no, id", (sid,)).fetchall(),
            key_lines=conn.execute(
                """SELECT c.name AS speaker, k.* FROM live_key_lines k LEFT JOIN characters c ON c.id = k.speaker_id
                   WHERE k.story_id = ? ORDER BY k.ep_no, k.id""", (sid,)).fetchall(),
            directives=directives(conn, sid, status=None), directive_history=history,
            summaries=[{**dict(r), "snapshot": json.loads(r["snapshot"])} for r in conn.execute(
                "SELECT * FROM summaries WHERE story_id = ? AND stale = 0 ORDER BY start_ep", (sid,))],
            feedback=conn.execute("SELECT * FROM feedback WHERE story_id = ? ORDER BY id", (sid,)).fetchall(),
        )

    @app.post("/stories/{sid}/directives/{did}")
    def set_directive(sid: int, did: int, status: str = Form(...), reason: str = Form(""),
                      conn: sqlite3.Connection = Depends(db)):
        if status not in ("active", "paused", "fulfilled", "superseded"):
            raise HTTPException(400, "unknown status")
        row = conn.execute("SELECT id FROM directives WHERE id = ? AND story_id = ?", (did, sid)).fetchone()
        if row is None:
            raise HTTPException(404)
        conn.execute("INSERT INTO directive_status (directive_id, status, reason) VALUES (?, ?, ?)",
                     (did, status, reason.strip() or None))
        return back(f"/stories/{sid}/memory#instructions")

    @app.get("/healthz")
    def healthz() -> Response:
        return Response("ok")

    return app


def _progress(conn: sqlite3.Connection, sid: int) -> dict[str, Any]:
    """Latest decisions and model spend, for the progress panel."""
    steps = conn.execute(
        "SELECT node, decision, ep_no, created_at FROM steps WHERE story_id = ? ORDER BY id DESC LIMIT 8", (sid,)
    ).fetchall()
    usage = conn.execute(
        """SELECT COUNT(*) AS calls, COALESCE(SUM(status <> 'ok'), 0) AS failed,
                  COALESCE(SUM(prompt_tokens), 0) AS tokens_in, COALESCE(SUM(completion_tokens), 0) AS tokens_out,
                  COALESCE(SUM(latency_ms), 0) / 1000 AS seconds
           FROM llm_calls WHERE story_id = ?""",
        (sid,),
    ).fetchone()
    episodes = conn.execute(
        """SELECT ev.* FROM episode_versions ev WHERE ev.story_id = ? AND ev.id = (
               SELECT id FROM episode_versions x WHERE x.story_id = ev.story_id AND x.ep_no = ev.ep_no
               ORDER BY (x.status = 'approved') DESC, x.version DESC LIMIT 1)
           ORDER BY ev.ep_no""",
        (sid,),
    ).fetchall()
    return {"steps": steps, "usage": usage, "episodes": episodes, "active": directives(conn, sid)}


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


def _dashboard(conn: sqlite3.Connection, sid: int) -> dict[str, Any]:
    """Tokens and latency for the side panel: totals, per step, and per recent episode."""
    total = conn.execute(
        """SELECT COUNT(*) AS calls, COALESCE(SUM(status <> 'ok'), 0) AS failed,
                  COALESCE(SUM(prompt_tokens), 0) AS tokens_in, COALESCE(SUM(completion_tokens), 0) AS tokens_out,
                  COALESCE(SUM(latency_ms), 0) AS ms
           FROM llm_calls WHERE story_id = ?""", (sid,)).fetchone()
    latencies = [r[0] for r in conn.execute(
        "SELECT latency_ms FROM llm_calls WHERE story_id = ? AND status = 'ok'", (sid,))]
    by_step = conn.execute(
        """SELECT node, COUNT(*) AS calls, SUM(prompt_tokens + completion_tokens) AS tokens,
                  CAST(AVG(latency_ms) AS INTEGER) AS avg_ms
           FROM llm_calls WHERE story_id = ? GROUP BY node ORDER BY tokens DESC""", (sid,)).fetchall()
    episodes = conn.execute(
        """SELECT ep_no, COUNT(*) AS calls, SUM(prompt_tokens + completion_tokens) AS tokens,
                  SUM(latency_ms) AS ms, SUM(status <> 'ok') AS failed
           FROM llm_calls WHERE story_id = ? AND ep_no IS NOT NULL
           GROUP BY ep_no ORDER BY ep_no DESC LIMIT 8""", (sid,)).fetchall()
    written = conn.execute(
        "SELECT COUNT(DISTINCT ep_no) FROM episode_versions WHERE story_id = ? AND status = 'approved'", (sid,)
    ).fetchone()[0]
    all_eps = conn.execute(
        "SELECT SUM(prompt_tokens + completion_tokens), SUM(latency_ms) FROM llm_calls "
        "WHERE story_id = ? AND ep_no IS NOT NULL", (sid,)).fetchone()
    return {
        "total": total, "by_step": by_step, "episodes": episodes, "written": written,
        "p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
        "avg_ep_tokens": int((all_eps[0] or 0) / written) if written else 0,
        "avg_ep_seconds": int((all_eps[1] or 0) / 1000 / written) if written else 0,
    }


def _lines(text: str) -> list[str]:
    return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    return f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
