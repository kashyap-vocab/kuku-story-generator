from __future__ import annotations

from dataclasses import replace

import pytest

from serial_writer.engine import Engine
from serial_writer.llm import LLMError
from serial_writer.plan_shape import act_spans, all_arc_spans, split
from serial_writer.plan_store import load_acts, load_arcs, load_beats

from .fakes import SchemaFake

TOTAL = 20


@pytest.fixture
def make_engine(settings):
    engines = []

    def make(client):
        e = Engine(replace(settings, episode_token_budget=10**9), client=client)
        engines.append(e)
        return e

    yield make
    for e in engines:
        e.close()


def _start(engine) -> tuple[int, dict]:
    sid = engine.create_story("A rider delivers to the dead.", total_episodes=TOTAL)
    return sid, engine.advance(sid)


def _plans(engine, sid):
    return engine.conn.execute(
        "SELECT id, version, status, created_by, parent_id FROM plan_versions WHERE story_id = ? ORDER BY version",
        (sid,),
    ).fetchall()


def test_split_covers_every_episode_once():
    acts = act_spans(200)
    assert [(a.start, a.end) for a in acts] == [(1, 40), (41, 80), (81, 120), (121, 160), (161, 200)]
    arcs = [s for spans in all_arc_spans(acts).values() for s in spans]
    assert len(arcs) == 20 and [s.no for s in arcs] == list(range(1, 21))
    assert [e for s in arcs for e in range(s.start, s.end + 1)] == list(range(1, 201))
    assert split(1, 7, 3) == [(1, 3), (4, 5), (6, 7)]


def test_planning_stops_for_human_review(make_engine):
    engine = make_engine(SchemaFake())
    sid, status = _start(engine)

    assert status["state"] == "waiting"
    assert status["waiting_for"]["kind"] == "plan_review"
    pv = status["waiting_for"]["plan_version_id"]
    assert len(load_acts(engine.conn, pv)) == 5
    assert len(load_arcs(engine.conn, pv)) == 5
    assert [b["ep_no"] for b in load_beats(engine.conn, pv)] == list(range(1, TOTAL + 1))
    report = engine.conn.execute("SELECT check_report FROM plan_versions WHERE id = ?", (pv,)).fetchone()[0]
    assert report is not None
    assert engine.conn.execute("SELECT status FROM stories WHERE id = ?", (sid,)).fetchone()[0] == "plan_review"


def test_approve_locks_plan_and_records_feedback(make_engine):
    engine = make_engine(SchemaFake())
    sid, status = _start(engine)
    pv = status["waiting_for"]["plan_version_id"]

    status = engine.advance(sid, {"action": "approve", "note": "looks good"})
    assert status["state"] == "waiting" and status["waiting_for"]["kind"] == "write_more"
    assert engine.conn.execute("SELECT status FROM plan_versions WHERE id = ?", (pv,)).fetchone()[0] == "approved"
    assert engine.conn.execute("SELECT status FROM bible_versions WHERE story_id = ?", (sid,)).fetchone()[0] == "approved"
    assert engine.conn.execute("SELECT status FROM stories WHERE id = ?", (sid,)).fetchone()[0] == "writing"
    fb = engine.conn.execute("SELECT action, text FROM feedback WHERE story_id = ?", (sid,)).fetchall()
    assert [tuple(r) for r in fb] == [("plan_approve", "looks good")]


def test_edit_makes_new_version_and_asks_again(make_engine):
    fake = SchemaFake()
    engine = make_engine(fake)
    sid, status = _start(engine)
    pv = status["waiting_for"]["plan_version_id"]
    calls_before = len(fake.calls)

    status = engine.advance(sid, {
        "action": "edit", "note": "sharper start",
        "changes": {"acts": {"1": {"title": "The Wrong Route"}}, "beats": {"3": {"hook": "The door is already open."}}},
    })

    assert status["state"] == "waiting"
    new_pv = status["waiting_for"]["plan_version_id"]
    assert new_pv != pv
    plans = _plans(engine, sid)
    assert [(p["status"], p["created_by"]) for p in plans] == [("superseded", "model"), ("draft", "human")]
    assert plans[1]["parent_id"] == pv
    assert load_acts(engine.conn, new_pv)[0]["title"] == "The Wrong Route"
    assert load_beats(engine.conn, new_pv, 3, 3)[0]["hook"] == "The door is already open."
    assert load_acts(engine.conn, pv)[0]["title"] != "The Wrong Route"
    assert set(fake.calls[calls_before:]) == {"ActReviewOut", "RulesCheckOut"}


def test_bad_edit_is_rejected_and_review_asked_again(make_engine):
    engine = make_engine(SchemaFake())
    sid, status = _start(engine)
    pv = status["waiting_for"]["plan_version_id"]

    status = engine.advance(sid, {"action": "edit", "changes": {"beats": {"2": {"characters": ["Nobody"]}}}})

    assert status["state"] == "waiting"
    assert "Nobody" in status["waiting_for"]["error"]
    assert status["waiting_for"]["plan_version_id"] == pv
    assert len(_plans(engine, sid)) == 1


def test_redo_arc_rebuilds_only_that_arc(make_engine):
    fake = SchemaFake()
    engine = make_engine(fake)
    sid, status = _start(engine)
    pv = status["waiting_for"]["plan_version_id"]
    old = {b["ep_no"]: b["beat"] for b in load_beats(engine.conn, pv)}
    calls_before = len(fake.calls)

    status = engine.advance(sid, {"action": "redo", "target": "arc", "no": 2, "note": "make it scarier"})

    new_pv = status["waiting_for"]["plan_version_id"]
    new = {b["ep_no"]: b["beat"] for b in load_beats(engine.conn, new_pv)}
    assert sorted(new) == list(range(1, TOTAL + 1))
    arc2 = range(5, 9)
    assert all(new[e] == old[e] for e in new if e not in arc2)
    assert {c for c in fake.calls[calls_before:] if c not in ("ActReviewOut", "RulesCheckOut")} == {"BeatsOut", "RepeatCheckOut"}
    assert fake.calls[calls_before:].count("BeatsOut") <= 2
    beats_req = [r for r in fake.requests[calls_before:] if r["response_format"]["json_schema"]["name"] == "BeatsOut"][0]
    assert "make it scarier" in beats_req["messages"][-1]["content"]


def test_resume_after_crash_does_not_redo_finished_work(make_engine, settings):
    fail = {"on": True}

    def crash_on_beats(value, kwargs):
        if fail["on"]:
            raise RuntimeError("server died")
        return value

    fake = SchemaFake(hooks={"BeatsOut": crash_on_beats})
    engine = make_engine(fake)
    sid = engine.create_story("A rider delivers to the dead.", total_episodes=TOTAL)
    with pytest.raises(RuntimeError, match="server died"):
        engine.advance(sid)
    assert engine.status(sid)["state"] == "paused"

    fail["on"] = False
    engine2 = make_engine(fake)
    calls_before = len(fake.calls)
    status = engine2.advance(sid)

    assert status["state"] == "waiting"
    assert "Bible" not in fake.calls[calls_before:] and "ActsOut" not in fake.calls[calls_before:]
    assert engine2.conn.execute("SELECT COUNT(*) FROM bible_versions").fetchone()[0] == 1
    assert engine2.conn.execute("SELECT COUNT(*) FROM plan_acts").fetchone()[0] == 5


def test_llm_errors_are_logged_and_story_can_continue(make_engine):
    attempts = {"n": 0}

    def flaky_acts(value, kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 3:
            value["acts"] = value["acts"][:1]
        return value

    engine = make_engine(SchemaFake(hooks={"ActsOut": flaky_acts}))
    sid = engine.create_story("A rider delivers to the dead.", total_episodes=TOTAL)
    with pytest.raises(LLMError):
        engine.advance(sid)
    bad = engine.conn.execute(
        "SELECT COUNT(*) FROM llm_calls WHERE node = 'plan_acts' AND status = 'invalid_output'"
    ).fetchone()[0]
    assert bad == 3
    assert engine.advance(sid)["state"] == "waiting"


def test_arcs_cannot_duplicate_threads_or_people(make_engine):
    def bible(value, kwargs):
        value["cast"][0]["name"] = "Sarah Vance"
        value["threads"][0].update(key="missing_ledger", title="The Missing Ledger")
        return value

    def arcs(value, kwargs):
        for arc in value["arcs"]:
            arc["new_threads"] = [
                {"key": "missing_ledger_2", "title": "The missing ledger!", "question": "again?"},
                {"key": "night_shift", "title": "Night Shift", "question": "Who works nights?"},
            ]
            arc["new_characters"] = [
                {**arc["new_characters"][0], "name": "Sarah Vance (Voice Only)"},
                {**arc["new_characters"][0], "name": "Officer Chen"},
            ]
        return value

    engine = make_engine(SchemaFake(hooks={"Bible": bible, "ArcsOut": arcs}))
    sid = engine.create_story("A rider delivers to the dead.", total_episodes=100)
    pv = engine.advance(sid)["waiting_for"]["plan_version_id"]

    keys = [r[0] for r in engine.conn.execute("SELECT key FROM plan_threads WHERE plan_version_id = ?", (pv,))]
    assert keys.count("missing_ledger") == 1 and "missing_ledger_2" not in keys
    assert keys.count("night_shift") == 1
    names = [r[0] for r in engine.conn.execute("SELECT name FROM characters")]
    assert "Sarah Vance (Voice Only)" not in names and names.count("Officer Chen") == 1
    skipped = engine.conn.execute(
        "SELECT COUNT(*) FROM steps WHERE decision IN ('skipped_new_thread', 'skipped_new_character')"
    ).fetchone()[0]
    assert skipped > 0


def test_thread_lifecycle_rules():
    from serial_writer.plan_store import thread_lifecycle

    problems, state = thread_lifecycle([(1, "open"), (3, "advance"), (5, "resolve")])
    assert problems == [] and state == "resolved"
    problems, state = thread_lifecycle([(2, "advance"), (4, "resolve"), (6, "advance"), (8, "resolve")])
    assert [(ep, sev) for ep, sev, _ in problems] == [(2, "must_fix"), (6, "must_fix"), (8, "must_fix")]
    problems, _ = thread_lifecycle([(11, "advance")], state="resolved")
    assert problems and "after it was resolved" in problems[0][2]


def test_repeated_plan_lines_get_one_repair(make_engine):
    def flag_first_as_repeat(value, kwargs):
        rows = value["rows"]
        first = rows[0]["ep_no"]
        if first > 1:
            rows[0].update(same_event=True, closest_ep=1, what_is_alike="package vanishes again")
        return value

    fake = SchemaFake(hooks={"RepeatCheckOut": flag_first_as_repeat})
    engine = make_engine(fake)
    _start(engine)

    repairs = [r for r in fake.requests if r["response_format"]["json_schema"]["name"] == "BeatsOut"
               and len(r["messages"]) == 4 and "repeats ep" in r["messages"][-1]["content"]]
    assert len(repairs) == 4
    assert "repeats ep 1 (package vanishes again)" in repairs[0]["messages"][-1]["content"]
    assert repairs[0]["messages"][2]["role"] == "assistant"
    logged = engine.conn.execute("SELECT COUNT(*) FROM steps WHERE node = 'plan_beats' AND decision = 'repair'").fetchone()[0]
    assert logged >= 4


def test_similarity_scores():
    from serial_writer.similarity import near_copies, overlap

    a = "Elias and Miller enter the Hardware Core, a cooling chamber that smells of burnt hair."
    reworded = "Miller and Elias walk into the cooling chamber of the Hardware Core; it smells like burnt hair."
    different = "Sarah finds her mother's diary in the flooded basement and reads the last page."
    assert overlap(a, a) == 1.0
    assert overlap(a, reworded) >= 0.5
    assert overlap(a, different) < 0.1
    assert near_copies([(12, reworded), (13, different)], [(4, a)]) == [(12, 4, round(overlap(reworded, a), 2))]


def test_copied_arc_is_caught_even_when_the_model_says_its_fine(make_engine):
    first_arc: list[dict] = []

    def copy_first_arc(value, kwargs):
        if not first_arc:
            first_arc.extend(value["beats"])
        else:
            value["beats"] = [dict(b) for b in first_arc]
        return value

    def says_all_different(value, kwargs):
        for r in value["rows"]:
            r["same_event"] = False
        return value

    fake = SchemaFake(hooks={"BeatsOut": copy_first_arc, "RepeatCheckOut": says_all_different})
    engine = make_engine(fake)
    _, status = _start(engine)

    repairs = [r for r in fake.requests if r["response_format"]["json_schema"]["name"] == "BeatsOut"
               and len(r["messages"]) == 4]
    assert any("Ep 5 is nearly a copy of ep 1" in r["messages"][-1]["content"] for r in repairs)
    fresh = [r for r in fake.requests if r["response_format"]["json_schema"]["name"] == "BeatsOut"
             and len(r["messages"]) == 2 and r["temperature"] == 0.9]
    assert len(fresh) == 4
    report = engine.conn.execute(
        "SELECT check_report FROM plan_versions WHERE id = ?", (status["waiting_for"]["plan_version_id"],)
    ).fetchone()[0]
    assert "nearly a copy" in report


def test_quotes_must_really_be_in_the_text():
    from serial_writer.similarity import contains_quote

    text = "Ping.\n\nA notification pops up. Priority Delivery.\n\nI squint. Triple the usual rate. \"Oaksview?\" I say. \"That place is a graveyard.\""
    assert contains_quote(text, "Priority Delivery. Triple the usual rate.")
    assert contains_quote(text, "'Oaksview?' I say. 'That place is a graveyard.'")
    assert contains_quote(text, "that place is a graveyard")
    assert not contains_quote(text, "Priority Delivery. Quadruple the usual rate.")
    assert not contains_quote(text, "The building burned in 1994.")
    assert not contains_quote(text, "  ")


def test_fresh_attempt_replaces_a_copy_the_repair_could_not_fix(make_engine):
    first_arc: list[dict] = []

    def copy_unless_fresh(value, kwargs):
        if not first_arc:
            first_arc.extend(value["beats"])
        elif kwargs["temperature"] != 0.9:
            value["beats"] = [dict(b) for b in first_arc]
        return value

    fake = SchemaFake(hooks={"BeatsOut": copy_unless_fresh})
    engine = make_engine(fake)
    _, status = _start(engine)

    report = engine.conn.execute(
        "SELECT check_report FROM plan_versions WHERE id = ?", (status["waiting_for"]["plan_version_id"],)
    ).fetchone()[0]
    assert "nearly a copy" not in report
    kept = engine.conn.execute("SELECT detail FROM steps WHERE decision = 'fresh_attempt'").fetchall()
    assert kept and all('"kept": "fresh"' in k[0] for k in kept)


def test_thread_labels_the_model_gets_wrong_are_tidied_by_code(make_engine):
    def resolve_then_advance(value, kwargs):
        beats = value["beats"]
        key = beats[0]["threads"][0]["key"] if beats[0]["threads"] else None
        if key:
            events = ["advance", "resolve", "advance", "advance"]
            for b, ev in zip(beats, events):
                b["threads"] = [{"key": key, "event": ev}]
        return value

    engine = make_engine(SchemaFake(hooks={"BeatsOut": resolve_then_advance}))
    _, status = _start(engine)
    beats = load_beats(engine.conn, status["waiting_for"]["plan_version_id"], 1, 4)
    assert [b["threads"][0]["event"] for b in beats] == ["open", "advance", "advance", "resolve"]
    assert engine.conn.execute("SELECT COUNT(*) FROM steps WHERE decision = 'tidied_thread_labels'").fetchone()[0] >= 1


def test_story_size_follows_the_number_of_episodes():
    from serial_writer.plan_shape import story_size

    short, mid, long = story_size(15), story_size(50), story_size(200)
    assert (short.acts, short.cast, short.threads, short.arc_extras) == (3, (3, 5), (2, 3), 0)
    assert (mid.acts, mid.arc_extras) == (5, 1) and mid.cast[0] < long.cast[0]
    assert (long.acts, long.cast, long.threads, long.arc_extras) == (5, (10, 12), (7, 9), 2)
    acts = act_spans(15)
    assert [(a.start, a.end) for a in acts] == [(1, 5), (6, 10), (11, 15)]


def test_short_story_gets_a_small_cast_and_one_arc_per_act(make_engine):
    fake = SchemaFake()
    engine = make_engine(fake)
    sid = engine.create_story("A rider delivers to the dead.", total_episodes=15)
    status = engine.advance(sid)
    pv = status["waiting_for"]["plan_version_id"]

    bible_req = next(r for r in fake.requests if r["response_format"]["json_schema"]["name"] == "Bible")
    schema = bible_req["response_format"]["json_schema"]["schema"]["properties"]
    assert (schema["cast"]["minItems"], schema["cast"]["maxItems"]) == (3, 5)
    assert (schema["threads"]["minItems"], schema["threads"]["maxItems"]) == (2, 3)
    assert "3-5 people" in bible_req["messages"][-1]["content"]
    assert len(load_acts(engine.conn, pv)) == 3
    assert [(a["start_ep"], a["end_ep"]) for a in load_arcs(engine.conn, pv)] == [(1, 5), (6, 10), (11, 15)]
    assert "ArcsOut" not in fake.calls
    assert [b["ep_no"] for b in load_beats(engine.conn, pv)] == list(range(1, 16))

    status = engine.advance(sid, {"action": "edit", "changes": {"acts": {"2": {"goal": "Everything goes wrong."}}}})
    arc2 = load_arcs(engine.conn, status["waiting_for"]["plan_version_id"])[1]
    assert arc2["goal"] == "Everything goes wrong."
