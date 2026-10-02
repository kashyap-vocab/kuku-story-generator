"""The one way the app talks to the model.

Every attempt is logged to `llm_calls` (tokens, time, errors, full prompt and
reply). Structured calls use vLLM's JSON-schema decoding and are checked
against a Pydantic model before anyone uses the result.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .config import Settings
from .tracing import CallRecord, episode_tokens_used, log_call, log_step

T = TypeVar("T", bound=BaseModel)

_TRANSIENT = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LLMError(RuntimeError):
    """The model could not produce a usable answer within the allowed attempts."""


class BudgetExceeded(RuntimeError):
    """This episode has used up its token budget."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    attempts: int


@dataclass
class CallContext:
    """Where a call belongs, for logging and budgets."""

    node: str
    story_id: int | None = None
    ep_no: int | None = None
    run_id: int | None = None
    budget_since: str | None = None
    budgeted: bool = True


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.client = client or openai.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        self._sleep = sleep
        self._db_lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, str]],
        ctx: CallContext,
        *,
        temperature: float = 0.8,
        max_tokens: int = 1500,
    ) -> LLMResult:
        """Free-text call, e.g. drafting an episode."""
        result, _ = self._call(messages, ctx, temperature, max_tokens, response_format=None, parse=None)
        return result

    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        ctx: CallContext,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> T:
        """Call that must return JSON matching `schema`. Bad output counts as a failed attempt."""
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        _, parsed = self._call(
            messages, ctx, temperature, max_tokens,
            response_format=response_format, parse=schema.model_validate_json,
        )
        return parsed


    def _call(
        self,
        messages: list[dict[str, str]],
        ctx: CallContext,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        parse: Callable[[str], Any] | None,
    ) -> tuple[LLMResult, Any]:
        messages = self._prepare(messages)
        attempts = self.settings.llm_max_retries + 1
        last_error = "no attempts made"

        for attempt in range(1, attempts + 1):
            self._check_budget(ctx)
            rec = CallRecord(
                node=ctx.node, attempt=attempt, model=self.settings.llm_model,
                temperature=temperature, max_tokens=max_tokens, messages=messages,
                status="ok", run_id=ctx.run_id, story_id=ctx.story_id, ep_no=ctx.ep_no,
            )
            kwargs: dict[str, Any] = dict(
                model=self.settings.llm_model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format

            started = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except _TRANSIENT as exc:
                rec.latency_ms = _ms_since(started)
                rec.status, rec.error = "error", f"{type(exc).__name__}: {exc}"
                self._log(rec)
                last_error = rec.error
                if attempt < attempts:
                    self._sleep(5 * 3 ** (attempt - 1))
                continue
            except openai.APIError as exc:
                rec.latency_ms = _ms_since(started)
                rec.status, rec.error = "error", f"{type(exc).__name__}: {exc}"
                self._log(rec)
                raise LLMError(rec.error) from exc

            rec.latency_ms = _ms_since(started)
            choice = resp.choices[0]
            text = choice.message.content or ""
            rec.response = text
            rec.finish_reason = choice.finish_reason
            if resp.usage is not None:
                rec.prompt_tokens = resp.usage.prompt_tokens or 0
                rec.completion_tokens = resp.usage.completion_tokens or 0

            parsed = None
            problem = None
            if choice.finish_reason == "length":
                problem = f"reply cut off at max_tokens={max_tokens}"
            elif parse is not None:
                try:
                    parsed = parse(text)
                except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                    problem = f"output did not match schema: {exc}"

            if problem is not None:
                rec.status, rec.error = "invalid_output", problem
                self._log(rec)
                last_error = problem
                continue

            self._log(rec)
            result = LLMResult(
                text=text, prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                latency_ms=rec.latency_ms, attempts=attempt,
            )
            return result, parsed

        with self._db_lock:
            log_step(
                self.conn, ctx.node, "gave_up", run_id=ctx.run_id, story_id=ctx.story_id,
                ep_no=ctx.ep_no, detail={"attempts": attempts, "last_error": last_error},
            )
        raise LLMError(f"{ctx.node}: no usable answer after {attempts} attempts ({last_error})")

    def _check_budget(self, ctx: CallContext) -> None:
        if ctx.story_id is None or ctx.ep_no is None or not ctx.budgeted:
            return
        with self._db_lock:
            used = episode_tokens_used(self.conn, ctx.story_id, ctx.ep_no, ctx.budget_since)
        budget = self.settings.episode_token_budget
        if used >= budget:
            with self._db_lock:
                log_step(
                    self.conn, ctx.node, "budget_exceeded", run_id=ctx.run_id,
                    story_id=ctx.story_id, ep_no=ctx.ep_no, detail={"used": used, "budget": budget},
                )
            raise BudgetExceeded(f"episode {ctx.ep_no} used {used} of {budget} tokens")

    def _log(self, rec: CallRecord) -> None:
        with self._db_lock:
            log_call(self.conn, rec)

    def _prepare(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not self.settings.llm_merge_system_prompt:
            return list(messages)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [dict(m) for m in messages if m["role"] != "system"]
        if system and rest and rest[0]["role"] == "user":
            rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
        elif system:
            rest.insert(0, {"role": "user", "content": system})
        return rest


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
