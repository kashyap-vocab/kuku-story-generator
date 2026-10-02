"""Where acts and arcs start and end. Decided by code, not the model, so the plan
always covers every episode exactly once."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ARC_LEN = 10


@dataclass(frozen=True)
class Span:
    no: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def __contains__(self, ep: int) -> bool:
        return self.start <= ep <= self.end


def split(start: int, end: int, parts: int) -> list[tuple[int, int]]:
    """Split start..end into `parts` near-equal runs; earlier runs take the remainder."""
    total = end - start + 1
    if not 1 <= parts <= total:
        raise ValueError(f"cannot split {total} episodes into {parts} parts")
    size, extra = divmod(total, parts)
    out, cur = [], start
    for i in range(parts):
        n = size + (1 if i < extra else 0)
        out.append((cur, cur + n - 1))
        cur += n
    return out


@dataclass(frozen=True)
class StorySize:
    """How big the story's moving parts are, from its length. A short story with a
    long story's cast and questions can't pay them off; a long one with a short
    story's runs out of material."""

    acts: int
    cast: tuple[int, int]
    major_min: int
    threads: tuple[int, int]
    # New subplots and new people each arc may add on top of the story rules.
    arc_extras: int


def story_size(total_episodes: int) -> StorySize:
    """10 eps: 3 acts, 3-4 people, 2-3 questions. 200 eps: 5 acts, 10-12 people, 7-9 questions."""
    f = min(total_episodes, 200) / 200
    cast, threads = round(3 + 8 * f), round(2 + 6 * f)
    return StorySize(
        # Five acts need at least four episodes each to set up and turn.
        acts=3 if total_episodes < 20 else 5,
        cast=(max(3, cast - 1), min(12, cast + 1)),
        major_min=2 if total_episodes < 20 else 3,
        threads=(max(2, threads - 1), min(10, threads + 1)),
        arc_extras=0 if total_episodes < 20 else (1 if total_episodes < 60 else 2),
    )


def act_spans(total_episodes: int, n_acts: int | None = None) -> list[Span]:
    n = min(n_acts or story_size(total_episodes).acts, total_episodes)
    return [Span(i + 1, s, e) for i, (s, e) in enumerate(split(1, total_episodes, n))]


def arc_spans(act: Span, first_arc_no: int, arc_len: int = DEFAULT_ARC_LEN) -> list[Span]:
    """Arcs of roughly `arc_len` episodes inside one act, numbered across the whole story."""
    n = max(1, round(act.size / arc_len))
    return [Span(first_arc_no + i, s, e) for i, (s, e) in enumerate(split(act.start, act.end, n))]


def all_arc_spans(acts: list[Span], arc_len: int = DEFAULT_ARC_LEN) -> dict[int, list[Span]]:
    out: dict[int, list[Span]] = {}
    next_no = 1
    for act in acts:
        out[act.no] = arc_spans(act, next_no, arc_len)
        next_no += len(out[act.no])
    return out
