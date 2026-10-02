# Serial story writer

An agentic system that plans and writes a 200-episode serial story from a one-line
premise, with a human approving and steering it along the way. Built with LangGraph,
SQLite, and any OpenAI-compatible model server (we use Gemma 4 12B on vLLM).

How it works: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Status:** planning, episode writing, review, feedback, resume and a live usage
panel all work in the web app. The demo output is in [demo/](demo/): the full
200-episode plan, 20 written episodes, and the human interventions with their effects.
Design reasoning and measured cost are in [DECISIONS.md](DECISIONS.md).

## What it does

| Requirement | Where |
|---|---|
| Plan all 200 episodes (acts, arcs, one line per episode, threads, character arcs) | Planner, then a plan check (code + model) |
| Hold up at episode 150 | Memory pack of fixed size from tables, summaries and lookups, not the whole story |
| Catch contradictions and repeats | Continuity, plan and repetition checks; every problem must quote the text |
| Approve or edit the plan | Plan page: edit any line, redo an act or arc with a note, approve |
| Review, edit or reject an episode | Episode page: approve, edit the text, or request changes |
| Feedback carries forward | Lasting instructions go into every later episode; story changes re-plan the arc |
| Stop and resume | Pause, continue later, or continue after a crash |
| Traceable and bounded | Every model call logged; revision, retry and token limits per episode |

## Setup (about 5 minutes)

Needs Python 3.10+ and a running model server.

```bash
git clone https://github.com/kashyap-vocab/kuku-story-generator.git
cd kuku-story-generator
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then set LLM_BASE_URL and LLM_MODEL
```

In `.env`, point at your model server, for example:

```
LLM_BASE_URL=http://localhost:9010/v1
LLM_MODEL=surya2
```

## Running it

```bash
python -m serial_writer.web --host 0.0.0.0 --port 8000
```

Or from the command line (same engine, same database):

```bash
python -m serial_writer.cli new "your premise" --episodes 200
python -m serial_writer.cli plan 1 --eps 1-20     # read the plan
python -m serial_writer.cli approve 1
python -m serial_writer.cli status 1              # also: continue, redo, usage
```

Open http://localhost:8000, enter a premise, the number of episodes, and how often
you want to stop and review. The number of characters, story threads and acts is
set from the length: a 15-episode story gets 3 acts and 3-5 characters, a
200-episode one 5 acts and 10-12. Planning takes about 1.5 minutes for 15 episodes
and 13 minutes for 200; the story page shows progress and moves on by itself.

**The plan.** Read it by episode, by story and characters, or by the problems the
checks found. Click **Edit** on anything, or **Redo** an act or arc with a note.
Your changes are saved together and the plan is checked again. When you're happy,
**Approve the plan** and say how many episodes to write.

**Each episode** opens when it's ready. At the bottom:

- **Approve and continue** writes the next one.
- **Request changes**: say what should change and where it applies: just this
  episode, this and every episode after it (kept as a standing instruction that
  every later episode is written with and checked against), or a change to the
  story itself (the rest of the arc is re-planned). Or let the model decide, and
  confirm its choice.
- **Edit the text myself**: your version becomes the episode, and what the story
  remembers is taken from it.

Below the episode: what the checks found, what the story will remember if you
approve, the plan line, every version, and exactly what the model was given.

**Stop and come back.** Writing stops at the episode you asked for, or after the
current one if you press **Pause**. **Continue writing** picks up from there. If the
server stops mid-episode, the story page shows **Continue**, which resumes from the
last finished step. The review setting can be changed any time.

**Usage panel.** Every story page has a panel on the right: tokens used, median
latency per call, tokens and time per episode against the limit, and retried
calls. The numbers come from the call log, so they are exact.

Run it in `tmux` or with `nohup ... &` so it keeps going after you disconnect.
One process only: stories run in background threads inside it.

## Settings

All in `.env` (see `.env.example`): `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`,
`DB_PATH`, `EPISODE_TOKEN_BUDGET` (default 150000 per episode), `MAX_REVISIONS`
(default 2), `LLM_MAX_RETRIES`. Any OpenAI-compatible server works.

## Layout

```
serial_writer/
  planner.py, plan_*.py     planning, plan check, plan edits
  writer.py, episode_*.py   outline, draft, checks, revise, memory updates
  memory.py                 memory pack, memory updates, summaries
  graph.py, engine.py       LangGraph flow, human stops, background runs
  llm.py, tracing.py        model client, call/step logging
  schema.sql                the story's database
  web/                      FastAPI + HTMX pages
docs/ARCHITECTURE.md        design and reasoning
tests/                      fake-model tests, no server needed
```

## Tests

```bash
python -m pytest -q
```

The tests use a fake model, so they don't need the server.
