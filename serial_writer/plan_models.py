"""Answer shapes for the planning calls, and for the human's plan decisions.

Where the model must name a character or thread, the shape only allows names
that exist (built per call with `one_of`). vLLM enforces this while generating,
so a misspelt name can't turn into a duplicate character.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

# ---------------------------------------------------------------- helpers


def one_of(values: list[str]) -> Any:
    """A type that only accepts one of `values`."""
    if not values:
        raise ValueError("one_of needs at least one value")
    return Literal[tuple(dict.fromkeys(values))]


def thread_key(text: str) -> str:
    """'The Missing Ledger' -> 'the_missing_ledger'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "thread"


# ---------------------------------------------------------------- story rules


class CastMember(BaseModel):
    name: str = Field(description="Full name as it will be used in the story")
    role: str = Field(description="Their part in the story, a few words")
    description: str = Field(description="Who they are: age, job, look, manner, how they talk")
    wants: str = Field(description="What they want most")
    secret: str = Field(description="What they hide, or 'none'")
    importance: Literal["major", "supporting", "minor"]
    arc: str = Field(description="How they change from start to end of the story")
    enters_around_ep: int = Field(ge=1, description="Episode where they first appear")


class StoryThread(BaseModel):
    key: str = Field(description="Short snake_case id, e.g. missing_ledger")
    title: str
    question: str = Field(description="The open question the reader wants answered")


class Bible(BaseModel):
    title: str
    logline: str = Field(description="The whole story in one or two sentences")
    genre: str
    tone: str
    style_guide: list[str] = Field(min_length=5, max_length=10)
    world_rules: list[str] = Field(min_length=3, max_length=10)
    hidden_truth: str = Field(
        description="What is really going on, fully explained. Only the writers know this."
    )
    cast: list[CastMember] = Field(min_length=6, max_length=12)
    threads: list[StoryThread] = Field(min_length=4, max_length=10)


# ---------------------------------------------------------------- plan levels


def bible_model(size: Any) -> type[BaseModel]:
    """The story rules, with as many people and questions as the story's length can carry."""
    return create_model(
        "Bible", __base__=Bible,
        cast=(list[CastMember], Field(min_length=size.cast[0], max_length=size.cast[1])),
        threads=(list[StoryThread], Field(min_length=size.threads[0], max_length=size.threads[1])),
    )


def acts_model(n_acts: int, names: list[str], keys: list[str]) -> type[BaseModel]:
    Name, Key = one_of(names), one_of(keys)
    Endpoint = create_model("Endpoint", name=(Name, ...), where_they_end=(str, ...))
    Act = create_model(
        "Act",
        title=(str, ...),
        goal=(str, Field(description="What this act is about and what changes")),
        reveals=(str, Field(description="What the reader learns about the hidden truth in this act")),
        turning_point=(str, Field(description="The event in the act's last episode that turns the story")),
        opens=(list[Key], Field(default_factory=list)),
        resolves=(list[Key], Field(default_factory=list)),
        character_endpoints=(list[Endpoint], Field(default_factory=list)),
    )
    return create_model("ActsOut", acts=(list[Act], Field(min_length=n_acts, max_length=n_acts)))


class NewThread(BaseModel):
    key: str = Field(description="Short snake_case id")
    title: str
    question: str


class NewCharacter(BaseModel):
    name: str
    role: str
    description: str
    wants: str
    secret: str
    importance: Literal["supporting", "minor"]


def arcs_model(n_arcs: int, names: list[str], max_new: int = 2) -> type[BaseModel]:
    Name = one_of(names)
    Arc = create_model(
        "Arc",
        title=(str, ...),
        goal=(str, Field(description="What this run of episodes is about and what changes")),
        turning_point=(str, Field(description="The event in the arc's last episode")),
        focus_characters=(list[Name], Field(min_length=1, max_length=5)),
        new_threads=(list[NewThread], Field(default_factory=list, max_length=max_new)),
        new_characters=(list[NewCharacter], Field(default_factory=list, max_length=max_new)),
    )
    return create_model("ArcsOut", arcs=(list[Arc], Field(min_length=n_arcs, max_length=n_arcs)))


def beats_model(n_beats: int, names: list[str], keys: list[str]) -> type[BaseModel]:
    Name = one_of(names)
    # With every thread resolved there is nothing left to pick.
    Key = one_of(keys) if keys else str
    max_moves = 4 if keys else 0
    Move = create_model("ThreadMove", key=(Key, ...), event=(Literal["open", "advance", "resolve"], ...))
    Beat = create_model(
        "Beat",
        # Lengths stop a lazy answer ("Ep 12") from passing as a plan line.
        beat=(str, Field(min_length=40, description="What must happen in this episode, 2-3 plain sentences")),
        hook=(str, Field(min_length=15, description="The cliffhanger the episode ends on")),
        characters=(list[Name], Field(min_length=1, max_length=6)),
        threads=(list[Move], Field(default_factory=list, max_length=max_moves)),
    )
    return create_model("BeatsOut", beats=(list[Beat], Field(min_length=n_beats, max_length=n_beats)))


# ---------------------------------------------------------------- human decisions


class CastEdit(BaseModel):
    role: str | None = None
    description: str | None = None
    wants: str | None = None
    secret: str | None = None
    arc: str | None = None


class BibleEdit(BaseModel):
    title: str | None = None
    logline: str | None = None
    tone: str | None = None
    hidden_truth: str | None = None
    style_guide: list[str] | None = None
    world_rules: list[str] | None = None
    cast: dict[str, CastEdit] = Field(default_factory=dict)


class ActEdit(BaseModel):
    title: str | None = None
    goal: str | None = None
    turning_point: str | None = None


class ArcEdit(BaseModel):
    title: str | None = None
    goal: str | None = None
    turning_point: str | None = None


class ThreadMoveEdit(BaseModel):
    key: str
    event: Literal["open", "advance", "resolve"]


class BeatEdit(BaseModel):
    beat: str | None = None
    hook: str | None = None
    characters: list[str] | None = None
    threads: list[ThreadMoveEdit] | None = None


class PlanChanges(BaseModel):
    bible: BibleEdit | None = None
    acts: dict[int, ActEdit] = Field(default_factory=dict)
    arcs: dict[int, ArcEdit] = Field(default_factory=dict)
    beats: dict[int, BeatEdit] = Field(default_factory=dict)


class PlanDecision(BaseModel):
    """What the reviewer sends back when the plan is waiting for them.

    approve: lock the plan and start writing.
    edit:    apply the reviewer's own changes, re-check, ask again.
    redo:    have the model rebuild part of the plan using the note.
             target 'bible' = everything, 'all' = acts and below,
             'act' = that act's arcs and plan lines, 'arc' = that arc's plan lines.
    """

    action: Literal["approve", "edit", "redo"]
    note: str | None = None
    changes: PlanChanges | None = None
    target: Literal["bible", "all", "act", "arc"] | None = None
    no: int | None = None


# ---------------------------------------------------------------- reviews
#
# A small model asked "list any problems" tends to answer "none". These shapes
# make it give one answer per item instead, so it has to look at each one.


def repetition_model(new_eps: list[int], last_ep: int) -> type[BaseModel]:
    Row = create_model(
        "RepeatRow",
        ep_no=(Literal[tuple(new_eps)], ...),
        closest_ep=(int, Field(ge=0, le=last_ep, description="Most similar other episode, 0 if none")),
        what_is_alike=(str, Field(description="What the two have in common, in a few words")),
        same_event=(bool, Field(description="True if it is the same event happening again with nothing new")),
    )
    n = len(new_eps)
    return create_model("RepeatCheckOut", rows=(list[Row], Field(min_length=n, max_length=n)))


def act_review_model(eps: list[int]) -> type[BaseModel]:
    Row = create_model(
        "ActReviewRow",
        ep_no=(Literal[tuple(eps)], ...),
        what_changes=(str, Field(description="What is different in the story after this episode, a few words")),
        issue=(Literal["none", "pacing", "logic", "character", "hook"], ...),
        note=(str, Field(description="What is wrong, or empty if issue is none")),
    )
    n = len(eps)
    return create_model("ActReviewOut", rows=(list[Row], Field(min_length=n, max_length=n)))


def rules_review_model(item_ids: list[str]) -> type[BaseModel]:
    Row = create_model(
        "RuleReviewRow",
        item=(one_of(item_ids), ...),
        conflicts_with=(str, Field(description="Exact words from another part of the rules that clash with it, or 'none'")),
        note=(str, Field(description="Why they can't both be true, or empty")),
    )
    n = len(item_ids)
    return create_model("RulesCheckOut", rows=(list[Row], Field(min_length=n, max_length=n)))
