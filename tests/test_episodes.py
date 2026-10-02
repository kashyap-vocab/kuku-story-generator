from __future__ import annotations

import json
from dataclasses import replace

import pytest

from serial_writer.engine import Engine
from serial_writer.memory import character_states, directives
from serial_writer.plan_store import load_beats

from .fakes import SchemaFake, episode_text

TOTAL = 20


def remember(value, kwargs):
    """Memory updates that quote the episode, plus one fact with a made-up quote."""
    for c in value["characters"]:
        c.update(status="alive", location="the door", quote="walks to door 3 and knocks")
    value["facts"] = [
        {"category": "world", "text": "Door 3 opens only after two knocks.", "quote": "knocks twice"},
        {"category": "world", "text": "The door is blue.", "quote": "the blue door creaks"},  # not in the text
    ]
    value["threads"] = [{**value["threads"][0], "event": "opened", "quote": "The door opens by itself"}]
    value["key_lines"] = []
    value["new_characters"] = []
    value["relationships"] = []
    value["one_line"] = "Leo knocks on every door."
    return value


@pytest.fixture
def make(settings):
    engines = []

    def build(hooks=None, review="every_episode", every_n=None):
        fake = SchemaFake(hooks={"text": episode_text, "EpisodeMemory": remember, **(hooks or {})})
        engine = Engine(replace(settings, episode_token_budget=10**9), client=fake)
        engines.append(engine)
        sid = engine.create_story("A rider delivers to the dead.", total_episodes=TOTAL,
                                  review_mode=review, every_n=every_n)
        status = engine.advance(sid)
        assert status["waiting_for"]["kind"] == "plan_review"
        return engine, fake, sid

    yield build
    for e in engines:
        e.close()


def _version(engine, vid):
    return engine.conn.execute("SELECT * FROM episode_versions WHERE id = ?", (vid,)).fetchone()


def _drafts_for(fake, ep):
    return [r for r in fake.requests if r.get("response_format") is None
            and f"Write episode {ep}:" in r["messages"][-1]["content"]]


def test_episode_is_written_checked_and_waits_for_review(make):
    engine, fake, sid = make()
    status = engine.advance(sid, {"action": "approve", "write_until": 2})

    w = status["waiting_for"]
    assert w["kind"] == "episode_review" and w["ep_no"] == 1
    v = _version(engine, w["version_id"])
    assert v["status"] == "in_review" and 400 <= v["word_count"] <= 700
    assert json.loads(v["check_report"])["problems"] == [] or all(
        p["severity"] != "must_fix" for p in json.loads(v["check_report"])["problems"])
    # The memory pack is saved with the episode, with the ids of what was in it.
    ctx = engine.conn.execute("SELECT * FROM episode_contexts WHERE id = ?", (v["context_id"],)).fetchone()
    assert "=== THE PLAN ===" in ctx["pack"] and ">>> THIS EPISODE" in ctx["pack"]
    assert set(json.loads(ctx["refs"])) >= {"facts", "states", "thread_events", "directives"}
    # Proposed memory is stored but not live until approved; the made-up quote was dropped.
    facts = engine.conn.execute("SELECT text FROM facts WHERE source_version_id = ?", (v["id"],)).fetchall()
    assert [f[0] for f in facts] == ["Door 3 opens only after two knocks."]
    assert engine.conn.execute("SELECT COUNT(*) FROM live_facts").fetchone()[0] == 0
    # Steps are traced.
    nodes = {r[0] for r in engine.conn.execute("SELECT DISTINCT node FROM llm_calls WHERE ep_no = 1")}
    assert {"outline", "draft", "check_continuity", "check_plan", "extract_memory"} <= nodes


def test_approve_makes_memory_live_and_it_reaches_the_next_episode(make):
    engine, fake, sid = make()
    w = engine.advance(sid, {"action": "approve", "write_until": 2})["waiting_for"]

    status = engine.advance(sid, {"action": "approve", "note": "good"})

    assert status["waiting_for"]["ep_no"] == 2
    assert _version(engine, w["version_id"])["status"] == "approved"
    assert engine.conn.execute("SELECT COUNT(*) FROM live_facts").fetchone()[0] == 1
    assert character_states(engine.conn, sid, 2)
    pack = engine.conn.execute(
        "SELECT pack FROM episode_contexts WHERE ep_no = 2 ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert "Door 3 opens only after two knocks." in pack and "LAST EPISODE IN FULL (ep 1" in pack


def test_stops_at_the_run_target_and_continues_when_told(make):
    engine, fake, sid = make(review="on_issues")
    status = engine.advance(sid, {"action": "approve", "write_until": 3})

    # Clean episodes are approved without stopping when the setting is "on issues".
    assert status["waiting_for"]["kind"] == "write_more" and status["waiting_for"]["next_ep"] == 4
    approved = engine.conn.execute("SELECT ep_no FROM episode_versions WHERE status = 'approved' ORDER BY ep_no").fetchall()
    assert [r[0] for r in approved] == [1, 2, 3]

    status = engine.advance(sid, {"until": 5})
    assert status["waiting_for"]["next_ep"] == 6


def test_resume_after_a_crash_mid_episode(make, settings):
    crash = {"on": True}

    def flaky(value, kwargs):
        if crash["on"]:
            raise RuntimeError("server died")
        return remember(value, kwargs)

    engine, fake, sid = make(hooks={"EpisodeMemory": flaky})
    with pytest.raises(RuntimeError, match="server died"):
        engine.advance(sid, {"action": "approve", "write_until": 1})
    assert engine.status(sid)["state"] == "paused"

    crash["on"] = False
    drafts_before = len(_drafts_for(fake, 1))
    status = engine.advance(sid)
    assert status["waiting_for"]["kind"] == "episode_review"
    # It picked up at memory extraction: the draft wasn't written again.
    assert len(_drafts_for(fake, 1)) == drafts_before


def test_must_fix_problems_get_two_revisions_then_go_to_the_human(make):
    def too_short(kwargs):
        return "Episode 1: Short\n\nToo short."

    engine, fake, sid = make(hooks={"text": too_short}, review="on_issues")
    status = engine.advance(sid, {"action": "approve", "write_until": 1})

    w = status["waiting_for"]
    assert w["kind"] == "episode_review"  # "on issues" mode, and it has some
    v = _version(engine, w["version_id"])
    assert v["revisions"] == 2
    assert any(p["kind"] == "length" for p in json.loads(v["check_report"])["problems"])
    assert len([r for r in fake.requests if r.get("response_format") is None]) == 3  # draft + 2 revisions


def test_reject_rewrites_with_the_reason(make):
    engine, fake, sid = make()
    w = engine.advance(sid, {"action": "approve", "write_until": 1})["waiting_for"]

    status = engine.advance(sid, {"action": "reject", "reason": "Leo would never knock, he barges in"})

    assert _version(engine, w["version_id"])["status"] == "rejected"
    assert status["waiting_for"]["version_id"] != w["version_id"]
    assert "Leo would never knock" in _drafts_for(fake, 1)[-1]["messages"][-1]["content"]
    fb = engine.conn.execute("SELECT action, text FROM feedback WHERE ep_no = 1").fetchall()
    assert ("reject", "Leo would never knock, he barges in") in [tuple(r) for r in fb]
    # An empty reason is refused and the review is asked again.
    status = engine.advance(sid, {"action": "reject", "reason": " "})
    assert status["waiting_for"]["kind"] == "episode_review" and "say why" in status["waiting_for"]["error"]


def test_human_edit_becomes_the_episode_and_its_memory(make):
    engine, fake, sid = make()
    w = engine.advance(sid, {"action": "approve", "write_until": 2})["waiting_for"]
    edited = episode_text({"messages": [{"content": "Write episode 1:"}]}).replace("knocks twice", "knocks twice, then waits")

    status = engine.advance(sid, {"action": "edit", "text": edited, "note": "tighter"})

    assert status["waiting_for"]["ep_no"] == 2
    v1 = engine.conn.execute("SELECT * FROM episode_versions WHERE ep_no = 1 AND status = 'approved'").fetchone()
    assert v1["created_by"] == "human" and "then waits" in v1["text"] and v1["parent_id"] == w["version_id"]
    # Memory came from the edited text, and the model's version dropped out.
    assert engine.conn.execute("SELECT COUNT(*) FROM live_facts WHERE source_version_id = ?", (v1["id"],)).fetchone()[0] == 1
    assert engine.conn.execute("SELECT COUNT(*) FROM live_facts WHERE source_version_id = ?",
                               (w["version_id"],)).fetchone()[0] == 0


def _sorted_as(kind):
    def hook(value, kwargs):
        value.update(kind=kind, instruction="Keep the romance slow: no kiss before episode 15.", instruction_kind="pacing")
        return value
    return {"FeedbackSort": hook}


def test_lasting_instruction_is_confirmed_then_reaches_every_later_episode(make):
    engine, fake, sid = make(hooks=_sorted_as("lasting_instruction"))
    engine.advance(sid, {"action": "approve", "write_until": 3})

    status = engine.advance(sid, {"action": "feedback", "text": "slow down the romance"})
    w = status["waiting_for"]
    assert w["kind"] == "feedback_confirm" and w["feedback"]["kind"] == "lasting_instruction"
    assert directives(engine.conn, sid) == []  # nothing saved until the human confirms

    status = engine.advance(sid, {"kind": "lasting_instruction", "instruction": w["feedback"]["instruction"]})
    # Not rewriting: back to reviewing the same episode.
    assert status["waiting_for"]["kind"] == "episode_review" and status["waiting_for"]["ep_no"] == 1
    assert [d["text"] for d in directives(engine.conn, sid)] == ["Keep the romance slow: no kiss before episode 15."]

    engine.advance(sid, {"action": "approve"})
    engine.advance(sid, {"action": "approve"})
    for ep in (2, 3):
        assert "no kiss before episode 15" in _drafts_for(fake, ep)[-1]["messages"][-1]["content"]
    # And the checker is asked whether the episode follows it.
    checks = [r for r in fake.requests if r.get("response_format")
              and r["response_format"]["json_schema"]["name"] == "PlanCheckOut"]
    assert "Follows the human instruction: Keep the romance slow" in checks[-1]["messages"][-1]["content"]
    # The feedback and the human's confirmation are both on record.
    fb = engine.conn.execute("SELECT classification FROM feedback WHERE action = 'note'").fetchone()[0]
    assert json.loads(fb)["suggested"]["kind"] == "lasting_instruction"


def test_human_can_correct_the_sorting_and_cancel(make):
    engine, fake, sid = make(hooks=_sorted_as("story_change"))
    engine.advance(sid, {"action": "approve", "write_until": 1})
    engine.advance(sid, {"action": "feedback", "text": "the dialogue in scene 2 is stiff"})

    status = engine.advance(sid, {"kind": "cancel"})
    assert status["waiting_for"]["kind"] == "episode_review"

    engine.advance(sid, {"action": "feedback", "text": "the dialogue in scene 2 is stiff"})
    status = engine.advance(sid, {"kind": "fix_episode", "instruction": "Make scene 2's dialogue sound natural."})
    # Corrected to "fix this episode": rewritten with the note, no lasting instruction, plan unchanged.
    assert status["waiting_for"]["kind"] == "episode_review"
    assert "Make scene 2's dialogue sound natural." in _drafts_for(fake, 1)[-1]["messages"][-1]["content"]
    assert directives(engine.conn, sid) == []
    assert engine.conn.execute("SELECT COUNT(*) FROM plan_versions WHERE story_id = ?", (sid,)).fetchone()[0] == 1


def test_story_change_replans_the_rest_of_the_arc(make):
    engine, fake, sid = make(hooks=_sorted_as("story_change"))
    pv = engine.advance(sid, {"action": "approve", "write_until": 2})["waiting_for"]
    old_pv = engine.conn.execute("SELECT id FROM plan_versions WHERE status = 'approved'").fetchone()[0]
    engine.advance(sid, {"action": "approve"})  # ep 1
    old = {b["ep_no"]: b["beat"] for b in load_beats(engine.conn, old_pv)}

    engine.advance(sid, {"action": "feedback", "text": "kill off the landlord"})
    status = engine.advance(sid, {"kind": "story_change", "instruction": "The landlord dies in episode 2."})

    new_pv = engine.conn.execute("SELECT id FROM plan_versions WHERE status = 'approved'").fetchone()[0]
    assert new_pv != old_pv
    new = {b["ep_no"]: b["beat"] for b in load_beats(engine.conn, new_pv)}
    # Arc 1 is episodes 1-4: episode 1 is written and kept, 2-4 re-planned, the rest untouched.
    assert new[1] == old[1] and all(new[e] != old[e] for e in (2, 3, 4))
    assert all(new[e] == old[e] for e in range(5, TOTAL + 1))
    reason = engine.conn.execute("SELECT reason FROM plan_versions WHERE id = ?", (new_pv,)).fetchone()[0]
    assert "The landlord dies in episode 2." in reason
    # It changes the plan, not the standing instructions; episode 2 is written again from the new plan.
    assert directives(engine.conn, sid) == []
    assert status["waiting_for"]["ep_no"] == 2
    v = _version(engine, status["waiting_for"]["version_id"])
    assert v["plan_version_id"] == new_pv


def test_pause_stops_after_the_current_episode(make):
    engine, fake, sid = make(review="on_issues")
    engine.request_stop(sid)  # asked while the plan is waiting: applies once writing starts
    # Starting a run clears an old stop request, so ask again from inside the run.
    original = fake.hooks["EpisodeMemory"]

    def stop_during_ep2(value, kwargs):
        if "=== EPISODE 2 ===" in kwargs["messages"][-1]["content"]:
            engine.request_stop(sid)
        return original(value, kwargs)

    fake.hooks["EpisodeMemory"] = stop_during_ep2
    status = engine.advance(sid, {"action": "approve", "write_until": 10})
    assert status["waiting_for"]["kind"] == "write_more" and status["waiting_for"]["next_ep"] == 3


def test_finished_arc_is_summarised_and_the_summary_feeds_later_episodes(make):
    engine, fake, sid = make(review="on_issues")
    # 20 episodes = 5 acts of 4 = arcs of 4 episodes. Write past the end of arc 1.
    engine.advance(sid, {"action": "approve", "write_until": 5})

    rows = engine.conn.execute("SELECT * FROM summaries WHERE story_id = ?", (sid,)).fetchall()
    assert [(r["start_ep"], r["end_ep"], r["stale"]) for r in rows] == [(1, 4, 0)]
    snap = json.loads(rows[0]["snapshot"])
    assert {"characters", "open_threads", "timeline", "key_lines", "human_review_points"} <= set(snap)
    assert [t["ep"] for t in snap["timeline"]] == [1, 2, 3, 4]  # filled by code from the episodes
    pack = engine.conn.execute("SELECT pack FROM episode_contexts WHERE ep_no = 5").fetchone()[0]
    assert "Eps 1-4:" in pack


def test_changing_an_episode_makes_its_arc_summary_stale(make):
    from serial_writer.db import transaction
    from serial_writer.memory import approve_version

    engine, fake, sid = make(review="on_issues")
    engine.advance(sid, {"action": "approve", "write_until": 5})
    old = engine.conn.execute("SELECT * FROM episode_versions WHERE ep_no = 2 AND status = 'approved'").fetchone()
    with transaction(engine.conn):
        new = engine.conn.execute(
            """INSERT INTO episode_versions (story_id, ep_no, version, parent_id, plan_version_id, text, word_count, created_by)
               VALUES (?, 2, 99, ?, ?, 'a human rewrite', 3, 'human')""",
            (sid, old["id"], old["plan_version_id"])).lastrowid
        approve_version(engine.conn, sid, new)

    assert engine.conn.execute("SELECT stale FROM summaries").fetchone()[0] == 1
    engine.advance(sid, {"until": 6})  # moving on rebuilds it first
    rows = engine.conn.execute("SELECT start_ep, end_ep, stale FROM summaries ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(1, 4, 1), (1, 4, 0)]  # the old one is kept, marked stale


def test_reviewer_can_say_where_a_change_applies_and_skip_the_confirm(make):
    engine, fake, sid = make()
    engine.advance(sid, {"action": "approve", "write_until": 2})
    sorts_before = sum(1 for c in fake.calls if c == "FeedbackSort")

    status = engine.advance(sid, {"action": "feedback", "kind": "lasting_instruction", "text": "More dialogue, less narration."})

    # No sorting call, no confirm stop: saved for good, and this episode rewritten with it.
    assert sum(1 for c in fake.calls if c == "FeedbackSort") == sorts_before
    assert status["waiting_for"]["kind"] == "episode_review" and status["waiting_for"]["ep_no"] == 1
    assert [d["text"] for d in directives(engine.conn, sid)] == ["More dialogue, less narration."]
    assert "More dialogue, less narration." in _drafts_for(fake, 1)[-1]["messages"][-1]["content"]
    statuses = [r[0] for r in engine.conn.execute("SELECT status FROM episode_versions WHERE ep_no = 1 ORDER BY id")]
    assert statuses[0] == "rejected" and statuses[-1] == "in_review"
