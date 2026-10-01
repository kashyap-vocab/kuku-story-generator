"""Where acts and arcs start and end. Decided by code, not the model, so the plan
always covers every episode exactly once."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ACTS = 5
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


def act_spans(total_episodes: int, n_acts: int = DEFAULT_ACTS) -> list[Span]:
    n = min(n_acts, total_episodes)
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
