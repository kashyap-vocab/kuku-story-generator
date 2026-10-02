"""A stand-in for the vLLM server: answers every structured call with a valid
instance of the requested JSON schema, so whole flows run without a model."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from typing import Any, Callable

from .conftest import make_response


# Shared across calls, so no two fake answers ever contain the same string.
_COUNTER = itertools.count(1)


def instance(schema: dict[str, Any], defs: dict[str, Any] | None = None, counter=None) -> Any:
    """Smallest valid value for a JSON schema. Strings are numbered so names and keys stay unique."""
    defs = schema.get("$defs", {}) if defs is None else defs
    counter = counter or _COUNTER
    if "$ref" in schema:
        return instance(defs[schema["$ref"].split("/")[-1]], defs, counter)
    if "anyOf" in schema:
        return instance(schema["anyOf"][0], defs, counter)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return {k: instance(v, defs, counter) for k, v in props.items()}
    if kind == "array":
        # Exactly minItems when given, otherwise one item (if allowed).
        n = schema.get("minItems", 1 if schema.get("maxItems", 1) > 0 else 0)
        return [instance(schema["items"], defs, counter) for _ in range(n)]
    if kind == "integer":
        return schema.get("minimum", 1)
    if kind == "number":
        return schema.get("minimum", 1.0)
    if kind == "boolean":
        return False
    n = next(counter)
    text = f"text{n} note{n} line{n}"  # unique words, so fake lines never look like copies
    while len(text) < schema.get("minLength", 0):
        text += f" more{n}"
    return text


class SchemaFake:
    """Fake OpenAI client. `hooks[name]` can override the answer (or raise) for a schema name;
    `hooks["text"]` answers free-text calls."""

    def __init__(self, hooks: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]] | None = None):
        self.hooks = hooks or {}
        self.calls: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        rf = kwargs.get("response_format")
        name = rf["json_schema"]["name"] if rf else "text"
        self.calls.append(name)
        self.requests.append(kwargs)
        if rf is None:
            return make_response(self.hooks["text"](kwargs) if "text" in self.hooks else "Some prose.")
        schema = rf["json_schema"]["schema"]
        value = instance(schema)
        if name in self.hooks:
            value = self.hooks[name](value, kwargs)
        return make_response(json.dumps(value))


def episode_text(kwargs) -> str:
    """A clean ~500-word episode for whichever episode the prompt asks for."""
    import re

    prompt = kwargs["messages"][-1]["content"]
    ep = re.findall(r"episode (\d+)", prompt.split("=== YOUR TASK ===")[-1], flags=re.I)[0]
    body = " ".join(f"Leo walks to door {i} and knocks twice." for i in range(70))
    return f"Episode {ep}: The Door\n\n{body}\n\nThe door opens by itself."
