"""Cheap word-overlap similarity, for catching copied or near-copied text.

The model is good at finding which earlier episode is most like a new one, but
unreliable at saying whether they are "the same". Code makes that call instead.
"""

from __future__ import annotations

import re

_STOP = set(
    "a an and are as at be been but by for from had has have he her hers him his i if in into is it its "
    "me my no not of on or our out over she so than that the their them then there they this to too up "
    "us was we were what when where which while who will with you your just about after again all also "
    "back before being both can could did does down each even every get gets got how more most much "
    "must now off once only other same some still such through under very way well would".split()
)

# Plan lines this alike are copies, whatever the model says.
COPY = 0.5
# The model's "closest episode" pair counts as a repeat from this overlap up.
REPEAT = 0.35


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.casefold()) if len(w) > 2 and w not in _STOP}


def overlap(a: str, b: str) -> float:
    """Share of distinct content words the two texts have in common (0-1)."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def near_copies(new: list[tuple[int, str]], earlier: list[tuple[int, str]], threshold: float = COPY) -> list[tuple[int, int, float]]:
    """For each new (ep, text), the earlier or earlier-new episode it nearly copies.
    Returns (ep, copied_ep, overlap) for each one at or above `threshold`."""
    out = []
    seen = list(earlier)
    for ep, text in new:
        best = max(((overlap(text, t), e) for e, t in seen), default=(0.0, 0))
        if best[0] >= threshold:
            out.append((ep, best[1], round(best[0], 2)))
        seen.append((ep, text))
    return out
