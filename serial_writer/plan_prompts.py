"""Prompts for the planning calls.

Stable text (the head-writer brief, the story rules) comes first in every
prompt so vLLM can reuse its work between calls (prefix caching).
"""

from __future__ import annotations

from typing import Any

from .plan_shape import Span, StorySize

HEAD_WRITER = """You are the head writer of a long-running serial story told in plain, everyday spoken English.
Your job is planning, not prose. A good plan for a long serial:
- has ONE fixed answer to every mystery, decided up front, so clues never contradict each other;
- pays off every thread it opens, or leaves it open on purpose;
- keeps every character true to who they are, and lets them change for a reason;
- never uses the same story beat twice: no second identical near-miss, reveal or confrontation;
- ends every episode on a hook that makes the listener need the next one.
Be concrete: names, places, objects, times. Avoid vague words like "mysterious events" or "things get complicated".
Answer only with the JSON asked for."""


def _messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": HEAD_WRITER}, {"role": "user", "content": user}]


def _note(note: str | None) -> str:
    if not note:
        return ""
    return f"\nTHE REVIEWER ASKED FOR THIS. IT OVERRIDES EVERYTHING ELSE:\n{note}\n"


# ---------------------------------------------------------------- rendering


def render_bible(b: dict[str, Any], *, with_truth: bool = True) -> str:
    lines = [
        f"TITLE: {b['title']}",
        f"LOGLINE: {b['logline']}",
        f"GENRE: {b['genre']} | TONE: {b['tone']}",
        "STYLE:", *[f"- {r}" for r in b["style_guide"]],
        "WORLD RULES (always true):", *[f"- {r}" for r in b["world_rules"]],
    ]
    if with_truth:
        lines += ["HIDDEN TRUTH (writers only; reveal slowly):", b["hidden_truth"]]
    lines.append("CAST:")
    for c in b["cast"]:
        lines.append(
            f"- {c['name']} ({c['importance']}, {c['role']}, from ep ~{c['enters_around_ep']}): "
            f"{c['description']} Wants: {c['wants']} Secret: {c['secret']} Arc: {c['arc']}"
        )
    lines.append("MAIN THREADS:")
    lines += [f"- [{t['key']}] {t['title']}: {t['question']}" for t in b["threads"]]
    return "\n".join(lines)


def render_acts(acts: list[dict[str, Any]]) -> str:
    out = []
    for a in acts:
        d = a.get("details") or {}
        out.append(
            f"ACT {a['act_no']} (ep {a['start_ep']}-{a['end_ep']}): {a['title']}\n"
            f"  Goal: {a['goal']}\n"
            f"  Reveals: {d.get('reveals', '')}\n"
            f"  Turning point (ep {a['end_ep']}): {a['turning_point']}\n"
            f"  Opens: {', '.join(d.get('opens', [])) or '-'} | Resolves: {', '.join(d.get('resolves', [])) or '-'}"
        )
    return "\n".join(out)


def render_arcs(arcs: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"ARC {a['arc_no']} (act {a['act_no']}, ep {a['start_ep']}-{a['end_ep']}): {a['title']}: {a['goal']} "
        f"Ends with: {a['turning_point']}"
        for a in arcs
    )


def render_beats(beats: list[dict[str, Any]]) -> str:
    out = []
    for b in beats:
        moves = ", ".join(f"{m['event']} {m['key']}" for m in b["threads"]) or "-"
        out.append(
            f"Ep {b['ep_no']}: {b['beat']} | Hook: {b['hook']} | "
            f"With: {', '.join(b['characters'])} | Threads: {moves}"
        )
    return "\n".join(out)


def render_thread_status(status: list[dict[str, Any]], at_ep: int) -> str:
    if not status:
        return "(no threads yet)"
    out = []
    for t in status:
        if t["state"] == "open":
            out.append(
                f"- [{t['key']}] OPEN since ep {t['opened']}, last moved ep {t['last']} "
                f"({at_ep - t['last']} episodes ago): {t['question']}"
            )
        elif t["state"] == "resolved":
            out.append(f"- [{t['key']}] RESOLVED in ep {t['last']}: {t['question']}")
        else:
            out.append(f"- [{t['key']}] NOT OPENED YET: {t['question']}")
    return "\n".join(out)


# ---------------------------------------------------------------- prompts


def bible_prompt(premise: str, total_episodes: int, note: str | None, size: StorySize) -> list[dict[str, str]]:
    return _messages(f"""Create the story rules for a {total_episodes}-episode serial. Each episode is 400-700 words.
{_note(note)}
PREMISE: {premise}

What to write:
- hidden_truth: what is REALLY going on, fully explained: who, what, when, why and how. Specific enough that any clue can be checked against it. The listener only learns it slowly.
- cast: {size.cast[0]}-{size.cast[1]} people with clearly different voices, at least {size.major_min} of them major (the main character is always major). That is the right number for {total_episodes} episodes: every person must matter to the ending, so don't add anyone the story can't use. In "description", say how each one talks. Give the episode each one arrives around (1-{total_episodes}).
- threads: the {size.threads[0]}-{size.threads[1]} big open questions that carry the story, each one answered by the end. Keys are short snake_case ids.
- style_guide: 5-10 concrete rules for how the prose should sound: point of view, tense, sentence length, how much dialogue. It must read like a person talking: everyday words, short sentences, no flowery description.
- world_rules: 3-10 things that must always stay true (how the strange part of the story works, its limits, its costs).

Before answering, make sure nothing contradicts anything else: the world rules must let the premise actually happen, and every thread and character must fit the hidden truth.""")


def acts_prompt(
    premise: str, bible: dict[str, Any], spans: list[Span], note: str | None
) -> list[dict[str, str]]:
    ranges = "\n".join(f"- Act {s.no}: episodes {s.start}-{s.end}" for s in spans)
    return _messages(f"""{render_bible(bible)}

PREMISE: {premise}
{_note(note)}
Split the story into exactly {len(spans)} acts with these fixed episode ranges:
{ranges}

For each act, in order:
- goal: what the act is about and what changes by its end.
- reveals: what the listener learns about the hidden truth during this act. Spread the reveals out so the mystery lasts to the final act, and never contradict the hidden truth.
- turning_point: the specific event in the act's LAST episode that turns the story in a new direction.
- opens / resolves: which main threads (by key) start or end in this act. Every thread should open somewhere, and the main mystery resolves in the last act.
- character_endpoints: where each major character stands at the end of the act.

Stakes must rise act by act. The last act pays off the hidden truth.""")


def arcs_prompt(
    bible: dict[str, Any],
    acts: list[dict[str, Any]],
    act_no: int,
    spans: list[Span],
    earlier_arcs: list[dict[str, Any]],
    later_arcs: list[dict[str, Any]],
    subplots: list[dict[str, Any]],
    extra_cast: list[str],
    note: str | None,
) -> list[dict[str, str]]:
    ranges = "\n".join(f"- Arc {s.no}: episodes {s.start}-{s.end}" for s in spans)
    known = ""
    if subplots:
        known += "SUBPLOTS ALREADY ADDED BY EARLIER ARCS:\n" + "\n".join(
            f"- [{t['key']}] {t['title']}: {t['question']}" for t in subplots) + "\n"
    if extra_cast:
        known += "PEOPLE ALREADY ADDED BY EARLIER ARCS: " + ", ".join(extra_cast) + "\n"
    earlier = render_arcs(earlier_arcs) if earlier_arcs else "(none, this is the start)"
    later = f"\nARCS ALREADY PLANNED AFTER THIS ACT (stay consistent with them):\n{render_arcs(later_arcs)}\n" if later_arcs else ""
    return _messages(f"""{render_bible(bible)}

ACT PLAN:
{render_acts(acts)}

ARCS ALREADY PLANNED BEFORE THIS ACT:
{earlier}
{known}{later}{_note(note)}
Now split ACT {act_no} into exactly {len(spans)} arcs with these fixed episode ranges:
{ranges}

For each arc:
- goal: what this run of episodes is about and what changes.
- turning_point: the event in the arc's last episode. The last arc's turning point IS the act's turning point.
- focus_characters: who drives it.
- new_threads: usually EMPTY. Only add a subplot that asks a question none of the main threads or existing subplots already ask. Never re-add an existing thread under a new key.
- new_characters: usually EMPTY. Only a new PERSON the arc can't work without. Not a place, an apartment, a device, or another version of someone already in the cast.

Each arc must feel different from the ones before it: a new place, a new problem, a new pressure. Don't recycle an earlier arc's shape.""")


def beats_prompt(
    bible: dict[str, Any],
    act: dict[str, Any],
    arc: dict[str, Any],
    prev_arc: dict[str, Any] | None,
    prev_beats: list[dict[str, Any]],
    next_arc: dict[str, Any] | None,
    thread_status: str,
    extra_threads: list[dict[str, Any]],
    must_close: list[dict[str, Any]],
    note: str | None,
) -> list[dict[str, str]]:
    prev = (
        # Only its last few lines: enough to continue from, too little to copy.
        f"PREVIOUS ARC {prev_arc['arc_no']}: {prev_arc['title']}: {prev_arc['goal']}\n"
        f"It ends like this (continue from here; NEVER copy or reuse these lines):\n{render_beats(prev_beats[-3:])}"
        if prev_arc else "(this is the first arc)"
    )
    nxt = (
        f"NEXT ARC (its events belong to it: set them up, but do NOT reach them in this arc): "
        f"{next_arc['title']}: {next_arc['goal']} It ends with: {next_arc['turning_point']}"
        if next_arc else "(this is the last arc of the story)"
    )
    subplots = (
        "SUBPLOTS ADDED BY ARCS:\n" + "\n".join(f"- [{t['key']}] {t['title']}: {t['question']}" for t in extra_threads)
        if extra_threads else ""
    )
    eps = list(range(arc["start_ep"], arc["end_ep"] + 1))
    turning = (
        f"\nEpisode {act['end_ep']} is the last episode of the act. It must deliver the act's turning point: {act['turning_point']}"
        if act["end_ep"] == arc["end_ep"] else ""
    )
    closing = (
        "\nTHESE THREADS MUST BE RESOLVED IN THIS ARC (answered for good):\n"
        + "\n".join(f"- [{t['key']}] {t['question']}" for t in must_close)
        if must_close else ""
    )
    return _messages(f"""{render_bible(bible)}
{subplots}

CURRENT ACT {act['act_no']} (ep {act['start_ep']}-{act['end_ep']}): {act['title']}: {act['goal']}
CURRENT ARC {arc['arc_no']} (ep {arc['start_ep']}-{arc['end_ep']}): {arc['title']}: {arc['goal']}
Arc ends with (ep {arc['end_ep']}): {arc['turning_point']}

{prev}

{nxt}

WHERE THE THREADS STAND BEFORE THIS ARC:
{thread_status}
{closing}{_note(note)}
Write the plan line for each of episodes {eps[0]}-{eps[-1]} ({len(eps)} episodes, in order).{turning}

For each episode:
- beat: what must happen, in 2-3 plain sentences. One clear event that moves the story. Not "tension builds".
- hook: the specific cliffhanger it ends on.
- characters: who is in it. Only people who have arrived in the story by then.
- threads: which threads this episode touches, by key:
  - "open": the question is raised for the first time.
  - "advance": we learn something new about it, but it is NOT answered yet.
  - "resolve": the question is answered FOR GOOD. Use this once per thread, ever. After that, never use the thread again.
  Only "open" a thread that is NOT OPENED YET. Never touch a RESOLVED thread.

No two episodes may have the same beat: no second identical vanishing, chase, warning, confrontation or reveal. Every episode must show something the listener hasn't seen before. Don't repeat anything from the previous arc. Keep long-open threads alive by touching them now and then.""")


def repetition_prompt(ledger: list[dict[str, Any]], new_beats: list[dict[str, Any]]) -> list[dict[str, str]]:
    earlier = "\n".join(f"Ep {b['ep_no']}: {b['beat']}" for b in ledger) or "(none)"
    new = "\n".join(f"Ep {b['ep_no']}: {b['beat']} | Hook: {b['hook']}" for b in new_beats)
    return _messages(f"""EARLIER EPISODES:
{earlier}

NEW EPISODES:
{new}

Check each NEW episode for repetition, one row per new episode:
- closest_ep: the episode (earlier, or another new one) most like it. 0 only if nothing is alike at all.
- what_is_alike: what the two have in common.
- same_event: true if it is basically the same event again with nothing new for the listener (another package vanishing the same way, another identical warning, the same kind of fight with the same result). False if it builds on the earlier one with something new.""")


def repair_messages(
    original: list[dict[str, str]], previous_answer: str, problems: list[str]
) -> list[dict[str, str]]:
    """Send the model its own answer back with the problems found, for one rewrite."""
    listed = "\n".join(f"- {p}" for p in problems)
    return [
        *original,
        {"role": "assistant", "content": previous_answer},
        {"role": "user", "content": f"""Your plan lines have these problems:
{listed}

Write the whole list again, for the same episodes. Fix every problem above. Keep everything that wasn't a problem as close to before as you can."""},
    ]


def act_review_prompt(
    bible: dict[str, Any], act: dict[str, Any], beats: list[dict[str, Any]], earlier: list[dict[str, Any]]
) -> list[dict[str, str]]:
    before = "\n".join(f"Ep {b['ep_no']}: {b['beat']}" for b in earlier[-15:]) or "(this is the start)"
    return _messages(f"""{render_bible(bible)}

JUST BEFORE THIS ACT:
{before}

ACT {act['act_no']} (ep {act['start_ep']}-{act['end_ep']}): {act['title']}: {act['goal']}
Turning point: {act['turning_point']}

PLAN LINES FOR THIS ACT:
{render_beats(beats)}

Review the act like a strict story editor, one row per episode, in order:
- what_changes: what is different in the story after this episode. If nothing really changes, say so.
- issue: the worst problem with this episode, or "none":
  - "pacing": nothing really changes, it just repeats the situation.
  - "logic": it breaks a world rule, the hidden truth, or something from an earlier episode; or someone is there before they arrive in the story.
  - "character": someone acts against who they are, with no reason given.
  - "hook": the hook is vague, or isn't really a cliffhanger.
- note: what exactly is wrong, naming the rule or episode it clashes with.""")


def rules_items(bible: dict[str, Any]) -> list[tuple[str, str]]:
    """The parts of the story rules to check one by one, with short ids."""
    items = [(f"W{i}", r) for i, r in enumerate(bible["world_rules"], 1)]
    items += [(f"T{i}", f"{t['title']}: {t['question']}") for i, t in enumerate(bible["threads"], 1)]
    items += [
        (f"C{i}", f"{c['name']}: {c['description']} Wants: {c['wants']} Secret: {c['secret']}")
        for i, c in enumerate(bible["cast"], 1)
    ]
    return items


def rules_review_prompt(bible: dict[str, Any]) -> list[dict[str, str]]:
    listed = "\n".join(f"[{i}] {text}" for i, text in rules_items(bible))
    return _messages(f"""{render_bible(bible)}

PARTS TO CHECK:
{listed}

Check each part above against everything else in the story rules, one row per part. Look hard for:
- two things that can't both be true (e.g. a fire in one place, an explosion in another);
- a world rule that would stop the premise itself from happening;
- a thread question that assumes something the hidden truth says is false.

conflicts_with: copy the EXACT words from another part of the story rules that clash with this part, or "none".
note: why they can't both be true.""")
