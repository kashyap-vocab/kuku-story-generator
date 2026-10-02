"""Prompts for writing, checking and remembering one episode.

The memory pack goes first in every call for an episode, so vLLM can reuse its
work across the outline, draft, checks and extraction (prefix caching).
"""

from __future__ import annotations

from typing import Any

MIN_WORDS, MAX_WORDS = 400, 700

WRITER = """You write a long-running audio serial in plain, everyday spoken English.
- Short sentences. Lots of dialogue. People talk like real people: interrupted, plain, a little messy.
- Show, don't explain. No summaries of feelings. No lectures.
- Every episode moves the story: something happens that can't be undone.
- End on a hook that makes the listener need the next episode.
- Never contradict the story rules, the facts already established, or what characters know.
- The human instructions override everything else."""

CLICHES = [
    "a chill that had nothing to do with", "couldn't help but", "little did", "testament to",
    "sent shivers down", "a wave of", "the air was thick with", "time seemed to stop", "heart pounded in",
    "breath he didn't know", "breath she didn't know", "a sense of dread", "eyes widened in",
    "the weight of the world", "it was as if", "something shifted", "in that moment", "unbeknownst",
]

CHECKER = """You check episodes of a long-running serial for mistakes before a human editor sees them.
Be strict but fair. Only report what you can prove by quoting the exact words. Answer only with the JSON asked for."""


def _with_pack(system: str, pack: str, task: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": f"{pack}\n\n=== YOUR TASK ===\n{task}"}]


def _must_fix(note: str | None) -> str:
    return f"\nTHE EDITOR REJECTED THE LAST VERSION. THIS MUST BE FIXED:\n{note}\n" if note else ""


def outline_prompt(pack: str, ep: int, note: str | None) -> list[dict[str, str]]:
    return _with_pack(WRITER, pack, f"""Plan episode {ep} as 3-5 scenes.
It must deliver this episode's plan line (marked >>> THIS EPISODE), follow the human instructions,
pick up where the last episode ended, and stop on a strong closing hook. Don't reach later episodes' events.
{_must_fix(note)}Answer with the JSON asked for.""")


def draft_prompt(pack: str, ep: int, outline: dict[str, Any], note: str | None) -> list[dict[str, str]]:
    scenes = "\n".join(
        f"{i}. {s['where']} ({', '.join(s['who'])}): {s['what_happens']} -> {s['what_changes']}"
        for i, s in enumerate(outline["scenes"], 1)
    )
    return _with_pack(WRITER, pack, f"""Write episode {ep}: "{outline['title']}".

SCENES:
{scenes}
ENDS ON: {outline['closing_hook']}
{_must_fix(note)}
Rules:
- {MIN_WORDS + 50}-{MAX_WORDS - 50} words.
- First line: Episode {ep}: {outline['title']}
- Then the episode itself. No notes, no "In this episode", no summary at the end.
- Never use these phrases: {"; ".join(CLICHES[:10])}.
- Stop on the hook. Don't resolve it.""")


def revise_prompt(pack: str, ep: int, text: str, problems: list[dict[str, Any]]) -> list[dict[str, str]]:
    listed = "\n".join(
        f"- {p['message']}" + (f' (the words: "{p["quote"]}")' if p.get("quote") else "") for p in problems
    )
    return _with_pack(WRITER, pack, f"""Here is a draft of episode {ep}:

{text}

An editor found these problems:
{listed}

Rewrite the episode to fix every problem above. Change only what's needed; keep everything else,
including the title line, the voice and the ending hook. {MIN_WORDS + 50}-{MAX_WORDS - 50} words.
Answer with the full episode only.""")


def continuity_prompt(pack: str, text: str) -> list[dict[str, str]]:
    return _with_pack(CHECKER, pack, f"""=== NEW EPISODE ===
{text}

Find places where the NEW EPISODE contradicts the memory above: a fact, where someone is,
who is dead, what someone knows or doesn't know yet, the timeline, or a world rule.
For each one, quote the exact words from the NEW EPISODE and the exact words from the memory it contradicts.
If there are none, return an empty list. Don't report style or anything you can't quote.""")


def plan_check_prompt(pack: str, text: str, items: list[tuple[str, str]], earlier: list[dict[str, Any]]) -> list[dict[str, str]]:
    listed = "\n".join(f"- {item}: {what}" for item, what in items)
    past = "\n".join(f"Ep {e['ep_no']}: {e['one_line']}" for e in earlier) or "(none)"
    return _with_pack(CHECKER, pack, f"""=== NEW EPISODE ===
{text}

Check the NEW EPISODE against each item, one row per item:
{listed}
For each: does the episode do it (yes / partly / no)? Quote the exact words that show it.

Then: which earlier episode is it most like? Earlier episodes:
{past}
same_event is true only if it's basically the same event again with nothing new for the listener.""")


def extraction_prompt(text: str, ep: int, cast: list[str], threads: list[dict[str, Any]]) -> list[dict[str, str]]:
    keys = "\n".join(f"- [{t['key']}] {t['title']}: {t['question']}" for t in threads) or "(none)"
    return [
        {"role": "system", "content": "You keep the record of a long-running serial. You only write down what the "
                                      "episode actually says, each with the exact words that show it."},
        {"role": "user", "content": f"""=== EPISODE {ep} ===
{text}

KNOWN CHARACTERS: {", ".join(cast)}
STORY QUESTIONS (threads):
{keys}

Write down what this episode adds to the story's memory:
- one_line and summary: what happened.
- story_time: the story day and time at the end of the episode.
- characters: one row for each known character who appears: alive or dead, where they end up, what they learned, what they want now.
- relationships: only if something between two people changed.
- facts: things that are now true and must stay true (what happened, objects, places, dates, rules of the world).
- threads: story questions this episode raises for the first time (opened), moves forward (advanced) or answers for good (resolved).
- key_lines: up to 4 lines worth calling back later (promises, threats, reveals), copied exactly.
- new_characters: named people who are not known characters.
Every quote must be copied exactly from the episode. Don't add anything the episode doesn't say."""},
    ]


def feedback_prompt(feedback: str, ep: int, plan_line: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You help the editor of a long-running serial sort their notes. Answer only with the JSON asked for."},
        {"role": "user", "content": f"""The editor read episode {ep} (plan: {plan_line}) and wrote:
"{feedback}"

Sort it:
- fix_episode: it's only about how this episode is written (a stiff line, a scene that drags).
- lasting_instruction: it's about how the story should be told from now on ("slow down the romance", "less gore", "more humour").
- story_change: something must happen in the story itself ("kill off Arjun", "Meena and Ravi break up").
Then rewrite it as one clear instruction for the writers."""},
    ]


def replan_prompt(
    bible_text: str, arc: dict[str, Any], written: list[dict[str, Any]], old: list[dict[str, Any]], change: str,
) -> list[dict[str, str]]:
    done = "\n".join(f"Episode {e['ep_no']}: {e['one_line']}" for e in written) or "(none yet)"
    planned = "\n".join(f"Episode {b['ep_no']}\n  what happens: {b['beat']}\n  hook: {b['hook']}" for b in old)
    return [
        {"role": "system", "content": "You are the head writer of a long-running serial. Answer only with the JSON asked for."},
        {"role": "user", "content": f"""{bible_text}

CURRENT ARC {arc['arc_no']} (ep {arc['start_ep']}-{arc['end_ep']}): {arc['title']}: {arc['goal']}
Arc ends with: {arc['turning_point']}

WHAT HAS ACTUALLY HAPPENED SO FAR:
{done}

THE OLD PLAN FOR THE REST OF THIS ARC:
{planned}

THE EDITOR HAS CHANGED THE STORY. THIS MUST HAPPEN AND IT OVERRIDES THE OLD PLAN:
{change}

Write new plan lines for episodes {old[0]['ep_no']}-{old[-1]['ep_no']}, one per episode, in order.
- Episode {old[0]['ep_no']} must make the change happen.
- Later episodes follow from it: nobody acts as if it didn't happen. Still reach the arc's ending, adjusted if needed.
- Keep what still works from the old plan.
For each episode:
- beat: what happens, in 2-3 full sentences (not just the episode number).
- hook: the cliffhanger it ends on.
- characters: who is in it.
- threads: story questions it opens, advances or resolves (may be empty)."""},
    ]


def arc_summary_prompt(arc: dict[str, Any], episodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    eps = "\n".join(f"Episode {e['ep_no']} ({e['story_time'] or 'time not noted'}): {e['summary'] or e['one_line']}"
                    for e in episodes)
    return [
        {"role": "system", "content": "You keep the record of a long-running serial. Plain, exact, no opinions."},
        {"role": "user", "content": f"""Arc {arc['arc_no']}, "{arc['title']}" (episodes {arc['start_ep']}-{arc['end_ep']}).

What happened, episode by episode:
{eps}

Write what happened in this arc in 120-180 words, in order, past tense.
Use names. Keep every death, reveal, promise and change in who knows what. End with where things stand.
Only what the episodes say. No headings, no commentary."""},
    ]
