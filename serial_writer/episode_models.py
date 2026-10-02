"""Answer shapes for the episode calls.

As in planning, names and thread keys can only be ones that exist (vLLM
enforces the list while generating), and every claim carries a quote that code
checks against the text.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from .plan_models import one_of


def outline_model(names: list[str]) -> type[BaseModel]:
    Scene = create_model(
        "Scene",
        where=(str, Field(description="A specific place")),
        who=(list[one_of(names)], Field(min_length=1, max_length=5)),
        what_happens=(str, Field(description="What happens, in 1-2 plain sentences")),
        what_changes=(str, Field(description="What is different at the end of the scene")),
    )
    return create_model(
        "Outline",
        title=(str, Field(description="Episode title, a few words")),
        scenes=(list[Scene], Field(min_length=3, max_length=5)),
        closing_hook=(str, Field(description="The exact situation the episode stops on")),
    )


def extraction_model(names: list[str], keys: list[str]) -> type[BaseModel]:
    Name = one_of(names)
    Key = one_of(keys) if keys else str
    quote = (str, Field(description="Exact words copied from the episode that show this"))
    Character = create_model(
        "CharacterUpdate",
        name=(Name, ...),
        status=(Literal["alive", "dead", "missing", "unknown"], ...),
        location=(str, Field(description="Where they are at the end of the episode")),
        learned=(list[str], Field(default_factory=list, max_length=3, description="New things they now know")),
        goal=(str, Field(description="What they want right now")),
        change=(str, Field(description="What changed for them in this episode, or 'nothing'")),
        quote=quote,
    )
    Relationship = create_model("RelationshipUpdate", a=(Name, ...), b=(Name, ...),
                                state=(str, Field(description="How things stand between them now")), quote=quote)
    Fact = create_model("NewFact", category=(Literal["timeline", "world", "object", "other"], ...),
                        text=(str, Field(description="One fact, as a plain sentence")), quote=quote)
    Thread = create_model("ThreadEvent", key=(Key, ...), event=(Literal["opened", "advanced", "resolved"], ...),
                          note=(str, Field(description="What happened to this question")), quote=quote)
    KeyLine = create_model("KeyLine", speaker=(one_of([*names, "narration"]), ...),
                           line=(str, Field(description="The exact line, copied word for word")),
                           why=(str, Field(description="Promise, threat, reveal or clue: why it matters later")))
    NewCharacter = create_model("NewCharacter", name=(str, ...), role=(str, ...), description=(str, ...),
                                location=(str, ...), quote=quote)
    return create_model(
        "EpisodeMemory",
        one_line=(str, Field(description="One sentence: what happened in this episode")),
        summary=(str, Field(description="About 100 words: what happened, in order")),
        story_time=(str, Field(description="Story day and time when the episode ends, e.g. 'Day 3, late night'")),
        characters=(list[Character], Field(default_factory=list, max_length=8,
                                           description="One row for each named character who appears in the episode")),
        relationships=(list[Relationship], Field(default_factory=list, max_length=5)),
        facts=(list[Fact], Field(default_factory=list, max_length=8)),
        threads=(list[Thread], Field(default_factory=list, max_length=5)),
        key_lines=(list[KeyLine], Field(default_factory=list, max_length=4)),
        new_characters=(list[NewCharacter], Field(default_factory=list, max_length=3,
                                                  description="Named people who are not in the cast list")),
    )


class ContinuityIssue(BaseModel):
    kind: Literal["contradiction", "wrong_place", "knows_too_much", "dead_character", "timeline", "world_rule"]
    quote: str = Field(description="Exact words from the NEW EPISODE that are wrong")
    clashes_with: str = Field(description="Exact words from the MEMORY or RULES it contradicts")
    note: str = Field(description="Why they can't both be true")


class ContinuityOut(BaseModel):
    issues: list[ContinuityIssue] = Field(default_factory=list, max_length=6)


def plan_check_model(items: list[str]) -> type[BaseModel]:
    """One row per item (the plan line, the hook, each human instruction), so the
    model has to look at each one instead of answering 'all fine'."""
    Row = create_model(
        "ItemCheck",
        item=(one_of(items), ...),
        verdict=(Literal["yes", "partly", "no"], Field(description="Does the episode do it?")),
        quote=(str, Field(description="Exact words from the episode that show your answer, or empty")),
        note=(str, Field(description="What is missing or wrong, or empty")),
    )
    n = len(items)
    return create_model(
        "PlanCheckOut",
        rows=(list[Row], Field(min_length=n, max_length=n)),
        closest_earlier_ep=(int, Field(ge=0, description="Earlier episode most like this one, 0 if none")),
        same_event=(bool, Field(description="True if this episode is basically that episode's event again")),
        what_is_alike=(str, Field(description="What the two have in common, or empty")),
    )


class FeedbackSort(BaseModel):
    kind: Literal["fix_episode", "lasting_instruction", "story_change"] = Field(
        description="fix_episode: only about this episode's writing. lasting_instruction: how the story should "
                    "be told from now on. story_change: something that must happen in the story itself")
    instruction: str = Field(description="The feedback rewritten as one clear instruction for the writers")
    instruction_kind: Literal["style", "pacing", "character", "plot", "other"]
    why: str = Field(description="Why you sorted it this way, one sentence")


def to_dict(model: Any) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else dict(model)
