from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from serial_writer.config import Settings
from serial_writer.db import connect, init_db


@pytest.fixture
def conn(tmp_path: Path):
    c = connect(tmp_path / "test.db")
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        llm_model="test-model",
        llm_max_retries=2,
        llm_merge_system_prompt=False,
        db_path=tmp_path / "test.db",
        episode_token_budget=1000,
    )


def make_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5,
                  finish_reason: str = "stop"):
    """Shape of an OpenAI chat completion, enough for LLMClient."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeOpenAI:
    """Returns (or raises) queued items in order and records each request."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
