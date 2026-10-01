# Serial story writer

An agentic system that plans and writes a 200-episode serial story from a one-line
premise, with a human approving and steering it along the way. Built with LangGraph,
SQLite, and any OpenAI-compatible model server (we use Gemma 4 12B on vLLM).

How it works: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Status:** planning stage done (story rules, 200-episode plan, plan check, plan
approval and redo, resume). Episode writing and the web app come next.

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

## Running a story

```bash
# Start a story and plan all 200 episodes (about 15-20 minutes)
python -m serial_writer.cli new "A delivery rider realizes every address on today's route belongs to someone who died in the same building."

# See the plan and the problems the checks found
python -m serial_writer.cli plan 1
python -m serial_writer.cli plan 1 --eps 1-20

# Approve it, or rebuild part of it with a note
python -m serial_writer.cli approve 1 --note "looks good"
python -m serial_writer.cli redo 1 --target arc --no 7 --note "make it scarier, no chases"

# If it stopped (crash, network, laptop asleep), pick up where it left off
python -m serial_writer.cli continue 1

# Where it is, and what it cost
python -m serial_writer.cli status 1
python -m serial_writer.cli usage 1
```

Long runs keep going after you disconnect if you start them in `tmux` or with
`nohup ... > plan.log 2>&1 &` on the server.

## Tests

```bash
python -m pytest -q
```

The tests use a fake model, so they don't need the server.
