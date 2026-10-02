from __future__ import annotations

import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from serial_writer.engine import Engine
from serial_writer.plan_store import load_acts, load_beats
from serial_writer.web.app import create_app

from .fakes import SchemaFake
from .test_episodes import _drafts_for

TOTAL = 20


@pytest.fixture
def web(settings):
    engine = Engine(replace(settings, episode_token_budget=10**9), client=SchemaFake())
    client = TestClient(create_app(engine), follow_redirects=False)
    yield engine, client
    _wait(engine, 1)
    engine.close()


def _wait(engine: Engine, sid: int) -> dict:
    """Background runs finish fast against the fake model."""
    for _ in range(500):
        status = engine.status(sid)
        if status["state"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("story still running")


def _new_story(engine, client) -> tuple[int, int]:
    r = client.post("/stories", data={"premise": "A rider delivers to the dead.", "episodes": TOTAL,
                                      "review_mode": "every_episode"})
    assert r.status_code == 303 and r.headers["location"] == "/stories/1"
    status = _wait(engine, 1)
    assert status["state"] == "waiting", engine.errors
    return 1, status["waiting_for"]["plan_version_id"]


def test_new_story_plans_in_background_and_shows_plan(web):
    engine, client = web
    sid, pv = _new_story(engine, client)

    page = client.get(f"/stories/{sid}")
    assert page.status_code == 200 and "Review the plan" in page.text
    plan = client.get(f"/stories/{sid}/plan")
    assert plan.status_code == 200
    assert "Approve the plan" in plan.text and "?tab=story" in plan.text
    assert all(f'id="ep-{ep}"' in plan.text for ep in range(1, TOTAL + 1))
    assert client.get("/").status_code == 200


def test_edits_are_collected_then_sent_as_one_new_version(web):
    engine, client = web
    sid, pv = _new_story(engine, client)
    beat = load_beats(engine.conn, pv, 3, 3)[0]

    r = client.post(f"/stories/{sid}/plan/edit/beat/3", data={
        "beat": "Leo finds the door already open.", "hook": "Someone is inside.",
        "characters": ", ".join(beat["characters"]),
        "threads": "\n".join(f"{m['event']} {m['key']}" for m in beat["threads"]),
    })
    assert r.status_code == 303
    act = load_acts(engine.conn, pv)[0]
    client.post(f"/stories/{sid}/plan/edit/acts/1", data={"title": "The Wrong Route", "goal": act["goal"],
                                                          "turning_point": act["turning_point"]})
    plan = client.get(f"/stories/{sid}/plan").text
    assert "2 changes not saved yet" in plan and ">changed<" in plan
    # Nothing is written until the edits are sent.
    assert load_beats(engine.conn, pv, 3, 3)[0]["beat"] == beat["beat"]
    # Approving with unsent edits is refused.
    assert "unsent" in client.post(f"/stories/{sid}/plan/approve", data={}).headers["location"]

    r = client.post(f"/stories/{sid}/plan/edits/send", data={"note": "sharper start"})
    assert r.headers["location"] == f"/stories/{sid}"
    status = _wait(engine, sid)
    new_pv = status["waiting_for"]["plan_version_id"]
    assert new_pv != pv
    assert load_beats(engine.conn, new_pv, 3, 3)[0]["beat"] == "Leo finds the door already open."
    assert load_acts(engine.conn, new_pv)[0]["title"] == "The Wrong Route"
    assert "not saved yet" not in client.get(f"/stories/{sid}/plan").text


def test_bad_edit_comes_back_as_an_error(web):
    engine, client = web
    sid, pv = _new_story(engine, client)
    r = client.post(f"/stories/{sid}/plan/edit/beat/2", data={"beat": "x", "threads": "explode the_key"})
    assert "error=" in r.headers["location"]

    client.post(f"/stories/{sid}/plan/edit/beat/2", data={"beat": "x", "characters": "Nobody"})
    client.post(f"/stories/{sid}/plan/edits/send", data={})
    status = _wait(engine, sid)
    # The graph refuses it and asks again; the page shows why.
    assert status["waiting_for"]["plan_version_id"] == pv
    assert "Nobody" in client.get(f"/stories/{sid}").text


def test_redo_needs_a_note_and_approve_starts_writing(web):
    engine, client = web
    sid, pv = _new_story(engine, client)
    r = client.post(f"/stories/{sid}/plan/redo", data={"target": "arc", "no": 2, "note": " "})
    assert "error=" in r.headers["location"]

    client.post(f"/stories/{sid}/plan/redo", data={"target": "arc", "no": 2, "note": "make it scarier"})
    status = _wait(engine, sid)
    assert status["waiting_for"]["plan_version_id"] != pv

    client.post(f"/stories/{sid}/plan/approve", data={"note": "good"})
    _wait(engine, sid)
    assert engine.conn.execute("SELECT status FROM stories WHERE id = ?", (sid,)).fetchone()[0] == "writing"
    # The approved plan is still viewable, without the review controls.
    plan = client.get(f"/stories/{sid}/plan").text
    assert "approved" in plan and "Approve the plan" not in plan
    assert client.post(f"/stories/{sid}/plan/approve", data={}).status_code == 409


@pytest.fixture
def web_writing(settings):
    from .fakes import episode_text
    from .test_episodes import _sorted_as, remember

    fake = SchemaFake(hooks={"text": episode_text, "EpisodeMemory": remember, **_sorted_as("lasting_instruction")})
    engine = Engine(replace(settings, episode_token_budget=10**9), client=fake)
    client = TestClient(create_app(engine), follow_redirects=False)
    yield engine, client, fake
    _wait(engine, 1)
    engine.close()


def test_episode_review_feedback_and_memory_pages(web_writing):
    engine, client, fake = web_writing
    sid, pv = _new_story(engine, client)
    client.post(f"/stories/{sid}/plan/approve", data={"write_until": 2})
    status = _wait(engine, sid)
    assert status["waiting_for"]["kind"] == "episode_review", engine.errors
    assert "Read episode 1" in client.get(f"/stories/{sid}").text

    page = client.get(f"/stories/{sid}/episodes/1").text
    assert "Approve and continue" in page and "What the story will remember if you approve" in page
    assert "Door 3 opens only after two knocks." in page and "What the model knew" in page

    # Feedback: sorted by the model, confirmed by the human, then it carries forward.
    client.post(f"/stories/{sid}/episodes/1/decide", data={"action": "feedback", "feedback": "slow down the romance"})
    assert _wait(engine, sid)["waiting_for"]["kind"] == "feedback_confirm"
    page = client.get(f"/stories/{sid}/episodes/1").text
    assert "How should your note be used?" in page and 'value="lasting_instruction" checked' in page
    client.post(f"/stories/{sid}/episodes/1/feedback", data={
        "kind": "lasting_instruction", "instruction": "No kissing yet.", "instruction_kind": "pacing"})
    assert _wait(engine, sid)["waiting_for"]["kind"] == "episode_review"
    assert "No kissing yet." in client.get(f"/stories/{sid}").text

    client.post(f"/stories/{sid}/episodes/1/decide", data={"action": "approve"})
    status = _wait(engine, sid)
    assert status["waiting_for"]["ep_no"] == 2
    assert "No kissing yet." in _drafts_for(fake, 2)[-1]["messages"][-1]["content"]
    # Deciding on an episode that isn't waiting is refused.
    assert "error=" in client.post(f"/stories/{sid}/episodes/1/decide", data={"action": "approve"}).headers["location"]

    client.post(f"/stories/{sid}/episodes/2/decide", data={"action": "reject", "reason": "Too calm."})
    assert _wait(engine, sid)["waiting_for"]["ep_no"] == 2
    client.post(f"/stories/{sid}/episodes/2/decide", data={"action": "approve"})
    status = _wait(engine, sid)
    assert status["waiting_for"]["kind"] == "write_more"
    assert "Write up to episode" in client.get(f"/stories/{sid}").text

    mem = client.get(f"/stories/{sid}/memory").text
    assert "No kissing yet." in mem and "Door 3 opens only after two knocks." in mem
    did = engine.conn.execute("SELECT id FROM directives").fetchone()[0]
    client.post(f"/stories/{sid}/directives/{did}", data={"status": "paused", "reason": "romance arc starts"})
    assert "paused" in client.get(f"/stories/{sid}/memory").text
    assert "No kissing yet." not in client.get(f"/stories/{sid}").text  # no longer a standing instruction

    # Old versions stay viewable.
    v1 = engine.conn.execute("SELECT id FROM episode_versions WHERE ep_no = 2 AND status = 'rejected'").fetchone()[0]
    assert "rejected" in client.get(f"/stories/{sid}/episodes/2?v={v1}").text

    client.post(f"/stories/{sid}/write", data={"until": 3})
    assert _wait(engine, sid)["waiting_for"]["ep_no"] == 3


def test_review_setting_can_change_any_time(web):
    engine, client = web
    sid, pv = _new_story(engine, client)
    client.post(f"/stories/{sid}/review", data={"review_mode": "every_n", "every_n": 3})
    rows = engine.conn.execute("SELECT mode, every_n FROM review_settings WHERE story_id = ? ORDER BY id", (sid,)).fetchall()
    assert [tuple(r) for r in rows] == [("every_episode", None), ("every_n", 3)]  # history kept
    assert 'value="every_n" selected' in client.get(f"/stories/{sid}").text
    assert "error=" in client.post(f"/stories/{sid}/review", data={"review_mode": "sometimes"}).headers["location"]


def test_request_changes_with_a_chosen_scope_and_edit_mode(web_writing):
    engine, client, fake = web_writing
    sid, pv = _new_story(engine, client)
    client.post(f"/stories/{sid}/plan/approve", data={"write_until": 2})
    _wait(engine, sid)

    # Edit mode shows the text in a box, with save and cancel.
    page = client.get(f"/stories/{sid}/episodes/1?edit=1").text
    assert 'name="text"' in page and "Save my version and approve" in page

    # "This and every episode after it": no confirm step, rewritten, kept for later episodes.
    client.post(f"/stories/{sid}/episodes/1/decide",
                data={"action": "feedback", "feedback": "Shorter scenes.", "kind": "lasting_instruction"})
    status = _wait(engine, sid)
    assert status["waiting_for"]["kind"] == "episode_review" and status["waiting_for"]["ep_no"] == 1
    assert "Shorter scenes." in client.get(f"/stories/{sid}").text


def test_status_poll_takes_the_reviewer_to_what_needs_them(web_writing):
    engine, client, fake = web_writing
    sid, pv = _new_story(engine, client)
    r = client.get(f"/stories/{sid}/status?was=running")
    assert r.headers.get("HX-Redirect") == f"/stories/{sid}/plan"
    client.post(f"/stories/{sid}/plan/approve", data={"write_until": 1})
    _wait(engine, sid)
    r = client.get(f"/stories/{sid}/status?was=running")
    assert r.headers.get("HX-Redirect") == f"/stories/{sid}/episodes/1"


def test_usage_dashboard_shows_tokens_and_latency(web):
    engine, client = web
    sid, _ = _new_story(engine, client)

    # Every story page carries the side panel, which loads the numbers itself.
    assert f'hx-get="/stories/{sid}/dashboard"' in client.get(f"/stories/{sid}").text
    panel = client.get(f"/stories/{sid}/dashboard")
    assert panel.status_code == 200
    assert "Tokens used" in panel.text and "Latency per call" in panel.text
    assert client.get("/stories/99/dashboard").status_code == 404
