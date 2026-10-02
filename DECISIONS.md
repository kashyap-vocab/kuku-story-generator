# Decisions

**Model.** Everything runs on a self-hosted open model: **Gemma 4 12B** (Google's pre-quantized 4-bit QAT release, w4a16) served with **vLLM** (48k context, 5 parallel requests, prefix caching). No API provider. A 12B model writes well but checks poorly when asked to do everything at once, so the design is **small jobs, small prompts, and code decides what the model sees.** The database is the memory; the model is only the writer.

## 1. How does it remember the story at episode 150?
Each episode gets a **memory pack of fixed size** (about the same at episode 5 and 150), built by code, not the full history:
- **Always in:** style, world rules, the **hidden truth** (so clues never contradict), and every active human instruction.
- **Plan:** current act and arc, this episode's plan line, 2 before and 3 after.
- **Summaries:** story so far, current act, latest arc. The "what happened" text is model-written; characters, timeline, open threads and key quotes are **filled by code from the tables**, so summaries of summaries cannot drift.
- **Recent:** last episode in full, one-liners for the 5 before it.
- **Looked up:** only the characters and threads in *this* plan line (alive or dead, location, what they know, thread history), plus older facts by keyword search (SQLite FTS5).

**Format:** plain labelled lines, not JSON: `Meena (major): dead since ep 12, at the lobby, knows: ...`; `[the_ledger] Who took it? opened ep 3, last moved ep 31 (overdue)`. Fewer tokens, and a small model reads it more reliably than nested JSON. JSON is used only for outputs we validate.

## 2. Where does the human step in?
(a) **The plan**, before any writing: a wrong turning point costs 200 episodes. (b) **Each episode**, at a rate they choose; problems the checks could not fix always stop. (c) **Feedback** is sorted into *this episode only*, a *lasting instruction* (in every later pack, checked and planned around), or a *story change* (arc re-planned). The human confirms the sorting, because a wrong guess silently steers the story. Not in between: memory updates are shown, not approved line by line.

## 3. Catching inconsistency and repetition before a human has to
Code checks (length, dead characters speaking, cliché list, format) then three focused model checks (continuity, plan and instructions, repetition). **Every problem must quote the exact words; code verifies the quote exists**, which stops a small model inventing problems. Memory updates need quotes too and can only name existing characters and threads. Two revisions maximum, then a human.

**Short stories don't repeat** because the planner is tuned to length: cast, threads and acts scale (15 episodes: 3 acts, 3–5 characters; 200: 5 acts, 10–12). Characters and threads are chosen from fixed lists, resolved threads can't be reused, characters can't appear before they arrive, each arc sees only a short slice of the previous one, a word-overlap check flags copied plan lines, and a fresh attempt replaces a repair that still copies.

## 4. Token optimization
Putting the whole story in context at episode 150 would be ~135k tokens, nearly 3x the 48k window. Instead:
- **Fixed-size pack** picked by code (2k at episode 1, about 4.4k by episode 20, capped by design), so cost per episode stays flat.
- **Summaries are code-filled**, only the narrative sentence is model-written.
- **Lookup without embeddings:** the plan line names the characters and threads, so we fetch exactly those rows; no embedding model or extra calls.
- **Free checks first:** length, dead-character and cliché checks cost no tokens and run before any model check.
- **Stable prefix:** rules go first and change last, so vLLM prefix caching reuses them between calls.
- **Capped output per call**, 3 attempts per call, 2 revisions, 150k-token ceiling per episode; checks return short fixed-schema answers (continuity check: ~2 output tokens when clean).
- **Parallel checks** across vLLM's 5 slots.

## 5. What breaks first, and the fix
1. **Drift between plan and story** after many human changes: an arc refresh before each arc re-reads the plan against what actually happened.
2. **Editing history** (rewrite episode 40): later episodes are not yet marked stale. Planned fix: every pack records the ids of the memory rows it used, so only episodes that depended on a changed row are flagged for revision.
3. **Keyword lookup** misses paraphrases; add embeddings.
4. **Voice flattening** over very long runs; add a style anchor excerpt.

## 6. Self-hosted vs Gemini
Measured on the demo story (200-episode plan, 20 episodes written, 2 feedback interventions, 0 failed calls):
- **Planning:** 57 calls, 310k tokens, 15 min.
- **Episodes:** 140 calls, 700k tokens, 20 min of model time, so **~35k tokens and ~61 s per approved episode**. That includes 10 rewritten versions caused by feedback and checks; a clean episode is ~25k tokens and ~45 s.
- **Memory pack:** 2.0k tokens at episode 1, 3.6k at 5, 4.4k at 15 to 20. It grows slowly and is capped by design; episode 150 is not yet measured.
- **200 episodes:** about 6.4M tokens in, 0.9M out, **~4 hours** end to end.

| | Gemma 4 12B, vLLM (measured) | Gemini Flash-class (estimate, no key) |
|---|---|---|
| Time per episode | ~61 s (~76 tok/s per request) | ~25-30 s (assumed 150-250 tok/s) |
| 200 episodes | ~4 h | ~2 h |
| Money | no per-token fee, own GPU | ~$4 (assumed $0.30/M in, $2.50/M out) |
| Data | stays in-house, no rate limits | leaves the building |
| Accuracy | not benchmarked | not benchmarked |

**Honest verdict:** we do not beat Gemini on speed, and at ~$4 per 200 episodes an API is no more expensive in money either. Self-hosting wins on control, privacy and free retries. We did not run a head-to-head on accuracy; the design assumes the weaker model and compensates with small jobs and quote-verified checks. 4-bit weights cut memory and speed up decoding at some cost in checking accuracy. One known weakness: the model cannot reliably shorten an episode on request, so 2 of 20 episodes ended slightly over 700 words after two revisions (accepted). Further savings: skip the outline call, re-check only what changed after a revision, shorter packs for quiet episodes.
