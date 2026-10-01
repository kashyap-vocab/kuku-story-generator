from dataclasses import replace

import httpx
import openai
import pytest
from pydantic import BaseModel

from serial_writer.llm import BudgetExceeded, CallContext, LLMClient, LLMError
from serial_writer.tracing import usage_by_node

from .conftest import FakeOpenAI, make_response

_REQ = httpx.Request("POST", "http://test/v1/chat/completions")


class Beat(BaseModel):
    ep_no: int
    beat: str


def _client(settings, conn, fake):
    return LLMClient(settings, conn, client=fake, sleep=lambda s: None)


def _calls(conn):
    return conn.execute("SELECT * FROM llm_calls ORDER BY id").fetchall()


def test_complete_logs_tokens_and_latency(settings, conn):
    fake = FakeOpenAI(make_response("Once upon a time", 120, 30))
    result = _client(settings, conn, fake).complete(
        [{"role": "user", "content": "write"}], CallContext("draft", ep_no=None)
    )
    assert result.text == "Once upon a time"
    (row,) = _calls(conn)
    assert (row["node"], row["status"], row["prompt_tokens"], row["completion_tokens"]) == ("draft", "ok", 120, 30)
    assert row["response"] == "Once upon a time"


def test_structured_sends_schema_and_parses(settings, conn):
    fake = FakeOpenAI(make_response('{"ep_no": 1, "beat": "Rider finds the first address"}'))
    beat = _client(settings, conn, fake).structured(
        [{"role": "user", "content": "plan"}], Beat, CallContext("plan_beats")
    )
    assert beat == Beat(ep_no=1, beat="Rider finds the first address")
    fmt = fake.requests[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"]["title"] == "Beat"


def test_bad_json_is_retried_and_every_attempt_logged(settings, conn):
    fake = FakeOpenAI(make_response('{"ep_no": "one"}'), make_response('{"ep_no": 1, "beat": "ok"}'))
    beat = _client(settings, conn, fake).structured([{"role": "user", "content": "x"}], Beat, CallContext("plan"))
    assert beat.beat == "ok"
    assert [r["status"] for r in _calls(conn)] == ["invalid_output", "ok"]


def test_cut_off_reply_is_retried(settings, conn):
    fake = FakeOpenAI(make_response("half a sto", finish_reason="length"), make_response("full story"))
    result = _client(settings, conn, fake).complete([{"role": "user", "content": "x"}], CallContext("draft"))
    assert result.text == "full story" and result.attempts == 2


def test_server_errors_retry_then_give_up(settings, conn):
    errors = [openai.APIConnectionError(request=_REQ) for _ in range(3)]
    fake = FakeOpenAI(*errors)
    with pytest.raises(LLMError, match="after 3 attempts"):
        _client(settings, conn, fake).complete([{"role": "user", "content": "x"}], CallContext("draft"))
    assert [r["status"] for r in _calls(conn)] == ["error"] * 3
    step = conn.execute("SELECT decision FROM steps").fetchone()
    assert step["decision"] == "gave_up"


def test_bad_request_is_not_retried(settings, conn):
    err = openai.BadRequestError("bad", response=httpx.Response(400, request=_REQ), body=None)
    fake = FakeOpenAI(err, make_response("never used"))
    with pytest.raises(LLMError):
        _client(settings, conn, fake).complete([{"role": "user", "content": "x"}], CallContext("draft"))
    assert len(_calls(conn)) == 1


def test_episode_budget_stops_further_calls(settings, conn):
    story = conn.execute("INSERT INTO stories (premise) VALUES ('p')").lastrowid
    ctx = CallContext("draft", story_id=story, ep_no=5)
    fake = FakeOpenAI(make_response("a", 900, 200), make_response("b"))
    client = _client(settings, conn, fake)
    client.complete([{"role": "user", "content": "x"}], ctx)  # uses 1100 of 1000
    with pytest.raises(BudgetExceeded):
        client.complete([{"role": "user", "content": "x"}], ctx)
    assert len(fake.requests) == 1
    # Another episode is unaffected.
    client.complete([{"role": "user", "content": "x"}], CallContext("draft", story_id=story, ep_no=6))


def test_system_prompt_merge(settings, conn):
    fake = FakeOpenAI(make_response("ok"))
    client = _client(replace(settings, llm_merge_system_prompt=True), conn, fake)
    client.complete(
        [{"role": "system", "content": "RULES"}, {"role": "user", "content": "write"}], CallContext("draft")
    )
    assert fake.requests[0]["messages"] == [{"role": "user", "content": "RULES\n\nwrite"}]


def test_usage_report(settings, conn):
    story = conn.execute("INSERT INTO stories (premise) VALUES ('p')").lastrowid
    fake = FakeOpenAI(make_response("a", 10, 5), make_response("b", 20, 5))
    client = _client(settings, conn, fake)
    client.complete([{"role": "user", "content": "x"}], CallContext("draft", story_id=story, ep_no=1))
    client.complete([{"role": "user", "content": "x"}], CallContext("draft", story_id=story, ep_no=2))
    (row,) = usage_by_node(conn, story)
    assert (row["node"], row["calls"], row["prompt_tokens"], row["completion_tokens"]) == ("draft", 2, 30, 10)
