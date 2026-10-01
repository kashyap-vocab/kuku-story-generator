"""Command line for running stories until the web app exists.

    python -m serial_writer.cli new "premise..."        start a story and plan it
    python -m serial_writer.cli continue 1              pick up after a stop or crash
    python -m serial_writer.cli status 1
    python -m serial_writer.cli plan 1 [--eps 1-20]     show the plan and its problems
    python -m serial_writer.cli approve 1 [--note ...]
    python -m serial_writer.cli redo 1 --target arc --no 7 --note "make it scarier"
    python -m serial_writer.cli usage 1                 tokens and time per step
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import Settings
from .engine import Engine
from .plan_store import load_acts, load_arcs, load_beats, load_bible
from .story import REVIEW_MODES
from .tracing import usage_by_node


def _print_status(status: dict) -> None:
    state = status["state"]
    if state == "waiting":
        w = status["waiting_for"]
        print(f"Waiting for you: {w['kind']} (plan version {w.get('plan_version_id')})")
        if w.get("error"):
            print(f"  Last decision could not be applied: {w['error']}")
    else:
        print(f"State: {state}" + (f" (next: {', '.join(status['next'])})" if status.get("next") else ""))


def _advance(engine: Engine, story_id: int, decision: dict | None = None) -> None:
    started = time.time()
    try:
        status = engine.advance(story_id, decision)
    except Exception as exc:
        print(f"Stopped after {time.time() - started:.0f}s: {type(exc).__name__}: {exc}")
        print(f"Nothing finished is lost. Run:  python -m serial_writer.cli continue {story_id}")
        sys.exit(1)
    print(f"Done in {time.time() - started:.0f}s.")
    _print_status(status)


def _plan_version(engine: Engine, story_id: int) -> int:
    status = engine.status(story_id)
    if status["state"] == "waiting" and status["waiting_for"].get("plan_version_id"):
        return status["waiting_for"]["plan_version_id"]
    row = engine.conn.execute(
        "SELECT id FROM plan_versions WHERE story_id = ? ORDER BY (status = 'approved') DESC, version DESC LIMIT 1",
        (story_id,),
    ).fetchone()
    if row is None:
        sys.exit("This story has no plan yet.")
    return row["id"]


def _show_plan(engine: Engine, story_id: int, eps: str | None) -> None:
    pv = _plan_version(engine, story_id)
    plan = engine.conn.execute("SELECT * FROM plan_versions WHERE id = ?", (pv,)).fetchone()
    bible = load_bible(engine.conn, plan["bible_version_id"])
    lo, hi = (int(x) for x in eps.split("-")) if eps else (1, 10**9)

    print(f"{bible['title']}  (plan v{plan['version']}, {plan['status']})\n{bible['logline']}\n")
    print("HIDDEN TRUTH:", bible["hidden_truth"], "\n")
    print("CAST:")
    for c in bible["cast"]:
        print(f"  - {c['name']} ({c['importance']}, from ep ~{c['enters_around_ep']}): {c['role']}")
    arcs = {a["arc_no"]: a for a in load_arcs(engine.conn, pv)}
    beats = load_beats(engine.conn, pv, lo, hi)
    for act in load_acts(engine.conn, pv):
        if act["end_ep"] < lo or act["start_ep"] > hi:
            continue
        print(f"\nACT {act['act_no']} (ep {act['start_ep']}-{act['end_ep']}): {act['title']}\n  {act['goal']}")
        for arc in (a for a in arcs.values() if a["act_no"] == act["act_no"]):
            if arc["end_ep"] < lo or arc["start_ep"] > hi:
                continue
            print(f"\n  ARC {arc['arc_no']} (ep {arc['start_ep']}-{arc['end_ep']}): {arc['title']}")
            for b in beats:
                if arc["start_ep"] <= b["ep_no"] <= arc["end_ep"]:
                    moves = ", ".join(f"{m['event']} {m['key']}" for m in b["threads"])
                    print(f"    {b['ep_no']:>3}. {b['beat']}\n         Hook: {b['hook']}"
                          + (f"\n         Threads: {moves}" if moves else ""))

    report = json.loads(plan["check_report"]) if plan["check_report"] else None
    if report:
        problems = report["problems"]
        must = [p for p in problems if p["severity"] == "must_fix"]
        print(f"\nPLAN CHECK: {len(problems)} problems ({len(must)} must fix)")
        for p in sorted(problems, key=lambda p: (p["severity"] != "must_fix", p["eps"] or [0])):
            eps_txt = f"ep {','.join(map(str, p['eps']))}" if p["eps"] else "-"
            print(f"  [{p['severity']}] {p['kind']} {eps_txt}: {p['message']}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m serial_writer.cli", description="Serial story writer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="start a story and plan it")
    p.add_argument("premise")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--review", choices=REVIEW_MODES, default="every_episode")
    p.add_argument("--every-n", type=int)

    for name, text in (("continue", "pick up after a stop or crash"), ("status", "where the story is"),
                       ("usage", "tokens and time per step")):
        sub.add_parser(name, help=text).add_argument("story_id", type=int)

    p = sub.add_parser("plan", help="show the plan and its problems")
    p.add_argument("story_id", type=int)
    p.add_argument("--eps", help="only these episodes, e.g. 1-20")

    p = sub.add_parser("approve", help="approve the plan")
    p.add_argument("story_id", type=int)
    p.add_argument("--note")

    p = sub.add_parser("redo", help="rebuild part of the plan with a note")
    p.add_argument("story_id", type=int)
    p.add_argument("--target", choices=["bible", "all", "act", "arc"], required=True)
    p.add_argument("--no", type=int, help="act or arc number")
    p.add_argument("--note", required=True)

    args = ap.parse_args(argv)
    engine = Engine(Settings.from_env())
    try:
        if args.cmd == "new":
            sid = engine.create_story(args.premise, total_episodes=args.episodes,
                                      review_mode=args.review, every_n=args.every_n)
            print(f"Story {sid} created. Planning (this takes a while)...")
            _advance(engine, sid)
        elif args.cmd == "continue":
            _advance(engine, args.story_id)
        elif args.cmd == "status":
            _print_status(engine.status(args.story_id))
        elif args.cmd == "plan":
            _show_plan(engine, args.story_id, args.eps)
        elif args.cmd == "approve":
            _advance(engine, args.story_id, {"action": "approve", "note": args.note})
        elif args.cmd == "redo":
            _advance(engine, args.story_id, {"action": "redo", "target": args.target, "no": args.no, "note": args.note})
        elif args.cmd == "usage":
            rows = usage_by_node(engine.conn, args.story_id)
            print(f"{'step':<20}{'calls':>6}{'failed':>7}{'tokens in':>11}{'tokens out':>11}{'seconds':>9}")
            for r in rows:
                print(f"{r['node']:<20}{r['calls']:>6}{r['failed']:>7}{r['prompt_tokens']:>11}"
                      f"{r['completion_tokens']:>11}{r['latency_ms'] / 1000:>9.0f}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
