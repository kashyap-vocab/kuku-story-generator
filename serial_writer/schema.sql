-- Serial story writer: database schema.
--
-- Ground rules (enforced below, not just by convention):
--   1. Nothing a human said or decided is ever deleted (feedback, directives, reviews).
--   2. Episode text is never overwritten. A change makes a new version.
--   3. Memory rows (character states, facts, threads, key lines) point at the
--      episode version they came from. They only count while that version is
--      approved. The live_* views apply this rule in one place, so a rejected or
--      replaced episode drops out of memory automatically.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- stories

CREATE TABLE IF NOT EXISTS stories (
    id              INTEGER PRIMARY KEY,
    premise         TEXT NOT NULL,
    title           TEXT,
    total_episodes  INTEGER NOT NULL DEFAULT 200 CHECK (total_episodes > 0),
    status          TEXT NOT NULL DEFAULT 'planning'
                    CHECK (status IN ('planning', 'plan_review', 'writing', 'paused', 'done')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- How often the human reviews. Append-only: the newest row is the current
-- setting, older rows are the history. Episodes the checker flags always go to
-- the human, whatever the mode.
CREATE TABLE IF NOT EXISTS review_settings (
    id                 INTEGER PRIMARY KEY,
    story_id           INTEGER NOT NULL REFERENCES stories(id),
    mode               TEXT NOT NULL
                       CHECK (mode IN ('every_episode', 'every_n', 'on_issues', 'arc_end')),
    every_n            INTEGER CHECK (every_n IS NULL OR every_n > 0),
    effective_from_ep  INTEGER NOT NULL DEFAULT 1,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (mode <> 'every_n' OR every_n IS NOT NULL)
);

-- ---------------------------------------------------------------- story rules

-- Story rules, world and style guide, as JSON. Versioned, never edited in place.
CREATE TABLE IF NOT EXISTS bible_versions (
    id          INTEGER PRIMARY KEY,
    story_id    INTEGER NOT NULL REFERENCES stories(id),
    version     INTEGER NOT NULL,
    content     TEXT NOT NULL CHECK (json_valid(content)),
    reason      TEXT,
    created_by  TEXT NOT NULL CHECK (created_by IN ('model', 'human')),
    status      TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'approved', 'superseded', 'rejected')),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (story_id, version)
);

CREATE TABLE IF NOT EXISTS characters (
    id                 INTEGER PRIMARY KEY,
    story_id           INTEGER NOT NULL REFERENCES stories(id),
    name               TEXT NOT NULL,
    role               TEXT,
    description        TEXT,
    importance         TEXT NOT NULL DEFAULT 'major'
                       CHECK (importance IN ('major', 'supporting', 'minor')),
    first_ep           INTEGER,
    -- NULL when created from the story rules; otherwise the episode that introduced them.
    source_version_id  INTEGER REFERENCES episode_versions(id),
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (story_id, name)
);

-- ---------------------------------------------------------------- the plan

-- Every re-plan makes a new plan version (a full copy of acts, arcs and beats),
-- so we can always see what the plan was and why it changed.
CREATE TABLE IF NOT EXISTS plan_versions (
    id          INTEGER PRIMARY KEY,
    story_id    INTEGER NOT NULL REFERENCES stories(id),
    version     INTEGER NOT NULL,
    parent_id   INTEGER REFERENCES plan_versions(id),
    bible_version_id INTEGER NOT NULL REFERENCES bible_versions(id),
    reason      TEXT,
    created_by  TEXT NOT NULL CHECK (created_by IN ('model', 'human')),
    status      TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'approved', 'superseded', 'rejected')),
    -- Problems found by the plan check (code + model), shown to the reviewer.
    check_report TEXT CHECK (check_report IS NULL OR json_valid(check_report)),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (story_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_approved_bible
    ON bible_versions (story_id) WHERE status = 'approved';

CREATE UNIQUE INDEX IF NOT EXISTS one_approved_plan
    ON plan_versions (story_id) WHERE status = 'approved';

CREATE TABLE IF NOT EXISTS plan_acts (
    id               INTEGER PRIMARY KEY,
    plan_version_id  INTEGER NOT NULL REFERENCES plan_versions(id),
    act_no           INTEGER NOT NULL,
    title            TEXT NOT NULL,
    goal             TEXT NOT NULL,
    turning_point    TEXT,
    start_ep         INTEGER NOT NULL,
    end_ep           INTEGER NOT NULL,
    -- Threads it opens and resolves, where each main character ends up.
    details          TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(details)),
    CHECK (start_ep <= end_ep),
    UNIQUE (plan_version_id, act_no)
);

CREATE TABLE IF NOT EXISTS plan_arcs (
    id               INTEGER PRIMARY KEY,
    plan_version_id  INTEGER NOT NULL REFERENCES plan_versions(id),
    arc_no           INTEGER NOT NULL,
    act_no           INTEGER NOT NULL,
    title            TEXT NOT NULL,
    goal             TEXT NOT NULL,
    turning_point    TEXT,
    start_ep         INTEGER NOT NULL,
    end_ep           INTEGER NOT NULL,
    -- Focus characters, and the threads and characters this arc adds.
    details          TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(details)),
    CHECK (start_ep <= end_ep),
    UNIQUE (plan_version_id, arc_no)
);

-- One line per episode: what must happen, who is in it, which threads it touches.
CREATE TABLE IF NOT EXISTS plan_beats (
    id                INTEGER PRIMARY KEY,
    plan_version_id   INTEGER NOT NULL REFERENCES plan_versions(id),
    ep_no             INTEGER NOT NULL,
    beat              TEXT NOT NULL,
    hook              TEXT NOT NULL DEFAULT '',
    characters        TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(characters)),
    threads           TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(threads)),
    is_turning_point  INTEGER NOT NULL DEFAULT 0 CHECK (is_turning_point IN (0, 1)),
    UNIQUE (plan_version_id, ep_no)
);

-- The plan's threads: the story's main questions plus subplots added by arcs.
-- Where each one opens and closes comes from the beats.
CREATE TABLE IF NOT EXISTS plan_threads (
    id               INTEGER PRIMARY KEY,
    plan_version_id  INTEGER NOT NULL REFERENCES plan_versions(id),
    key              TEXT NOT NULL,
    title            TEXT NOT NULL,
    question         TEXT NOT NULL,
    -- 'bible', or 'arc:<n>' for a subplot added by that arc.
    source           TEXT NOT NULL,
    UNIQUE (plan_version_id, key)
);

-- Who is in this plan's cast, and how each of them changes over the story.
CREATE TABLE IF NOT EXISTS plan_character_arcs (
    id               INTEGER PRIMARY KEY,
    plan_version_id  INTEGER NOT NULL REFERENCES plan_versions(id),
    character_id     INTEGER NOT NULL REFERENCES characters(id),
    arc              TEXT NOT NULL,
    -- 'bible', or 'arc:<n>' for a character added by that arc.
    source           TEXT NOT NULL DEFAULT 'bible',
    UNIQUE (plan_version_id, character_id)
);

-- ---------------------------------------------------------------- episodes

CREATE TABLE IF NOT EXISTS episode_versions (
    id               INTEGER PRIMARY KEY,
    story_id         INTEGER NOT NULL REFERENCES stories(id),
    ep_no            INTEGER NOT NULL,
    version          INTEGER NOT NULL,
    parent_id        INTEGER REFERENCES episode_versions(id),
    plan_version_id  INTEGER REFERENCES plan_versions(id),
    text             TEXT NOT NULL,
    word_count       INTEGER NOT NULL,
    -- One line saying what happened. Used to catch repeated plot beats.
    one_line         TEXT,
    created_by       TEXT NOT NULL CHECK (created_by IN ('model', 'human')),
    status           TEXT NOT NULL DEFAULT 'draft'
                     CHECK (status IN ('draft', 'in_review', 'approved', 'rejected', 'superseded')),
    check_report     TEXT CHECK (check_report IS NULL OR json_valid(check_report)),
    revisions        INTEGER NOT NULL DEFAULT 0,
    -- What the model was given when it wrote this version (see episode_contexts).
    context_id       INTEGER REFERENCES episode_contexts(id),
    -- ~100 words on what happened, and the story day and time it ends at. From memory extraction.
    summary          TEXT,
    story_time       TEXT,
    -- The scene outline the draft was written from.
    outline          TEXT CHECK (outline IS NULL OR json_valid(outline)),
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    decided_at       TEXT,
    UNIQUE (story_id, ep_no, version)
);

-- The memory pack an episode was written from, and the ids of every memory row
-- in it. Answers "what did the model know when it wrote episode 150?", and says
-- which later episodes depend on a fact when an earlier episode changes.
CREATE TABLE IF NOT EXISTS episode_contexts (
    id               INTEGER PRIMARY KEY,
    story_id         INTEGER NOT NULL REFERENCES stories(id),
    ep_no            INTEGER NOT NULL,
    plan_version_id  INTEGER NOT NULL REFERENCES plan_versions(id),
    pack             TEXT NOT NULL,
    refs             TEXT NOT NULL CHECK (json_valid(refs)),
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- At most one approved version of each episode, ever.
CREATE UNIQUE INDEX IF NOT EXISTS one_approved_episode
    ON episode_versions (story_id, ep_no) WHERE status = 'approved';

-- ---------------------------------------------------------------- memory

CREATE TABLE IF NOT EXISTS character_states (
    id                 INTEGER PRIMARY KEY,
    character_id       INTEGER NOT NULL REFERENCES characters(id),
    ep_no              INTEGER NOT NULL,
    source_version_id  INTEGER REFERENCES episode_versions(id),
    status             TEXT NOT NULL CHECK (status IN ('alive', 'dead', 'missing', 'unknown')),
    location           TEXT,
    knows              TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(knows)),
    goal               TEXT,
    notes              TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS relationships (
    id                 INTEGER PRIMARY KEY,
    story_id           INTEGER NOT NULL REFERENCES stories(id),
    char_a             INTEGER NOT NULL REFERENCES characters(id),
    char_b             INTEGER NOT NULL REFERENCES characters(id),
    ep_no              INTEGER NOT NULL,
    source_version_id  INTEGER REFERENCES episode_versions(id),
    state              TEXT NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- Store each pair one way round so lookups never miss.
    CHECK (char_a < char_b)
);

CREATE TABLE IF NOT EXISTS facts (
    id                 INTEGER PRIMARY KEY,
    story_id           INTEGER NOT NULL REFERENCES stories(id),
    ep_no              INTEGER,
    source_version_id  INTEGER REFERENCES episode_versions(id),
    category           TEXT NOT NULL CHECK (category IN ('timeline', 'world', 'object', 'other')),
    story_time         TEXT,
    text               TEXT NOT NULL,
    -- When a later fact replaces this one, it points back here instead of deleting it.
    replaced_by        INTEGER REFERENCES facts(id),
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS threads (
    id                 INTEGER PRIMARY KEY,
    story_id           INTEGER NOT NULL REFERENCES stories(id),
    -- The plan thread it follows (plan_threads.key).
    key                TEXT,
    title              TEXT NOT NULL,
    description        TEXT,
    opened_ep          INTEGER,
    source_version_id  INTEGER REFERENCES episode_versions(id),
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- A thread's current status is its latest event, so the history is never lost.
CREATE TABLE IF NOT EXISTS thread_events (
    id                 INTEGER PRIMARY KEY,
    thread_id          INTEGER NOT NULL REFERENCES threads(id),
    ep_no              INTEGER NOT NULL,
    source_version_id  INTEGER REFERENCES episode_versions(id),
    kind               TEXT NOT NULL
                       CHECK (kind IN ('opened', 'advanced', 'resolved', 'reopened', 'abandoned')),
    note               TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Exact lines worth calling back to later: promises, threats, reveals.
CREATE TABLE IF NOT EXISTS key_lines (
    id                 INTEGER PRIMARY KEY,
    story_id           INTEGER NOT NULL REFERENCES stories(id),
    ep_no              INTEGER NOT NULL,
    source_version_id  INTEGER REFERENCES episode_versions(id),
    speaker_id         INTEGER REFERENCES characters(id),
    line               TEXT NOT NULL,
    why                TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Summaries of a span of episodes. `narrative` is written by the model;
-- `snapshot` (characters, timeline, threads, main lines, human review points)
-- is filled from the tables above by code, so it cannot drift.
CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY,
    story_id    INTEGER NOT NULL REFERENCES stories(id),
    level       TEXT NOT NULL CHECK (level IN ('arc', 'act', 'story')),
    start_ep    INTEGER NOT NULL,
    end_ep      INTEGER NOT NULL,
    narrative   TEXT NOT NULL,
    snapshot    TEXT NOT NULL CHECK (json_valid(snapshot)),
    -- Set when an episode inside the span changes; the summary gets rebuilt.
    stale       INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------- the human

-- Every human action, in order. This is the record of human review points.
CREATE TABLE IF NOT EXISTS feedback (
    id                  INTEGER PRIMARY KEY,
    story_id            INTEGER NOT NULL REFERENCES stories(id),
    ep_no               INTEGER,
    episode_version_id  INTEGER REFERENCES episode_versions(id),
    plan_version_id     INTEGER REFERENCES plan_versions(id),
    action              TEXT NOT NULL CHECK (action IN (
                            'plan_approve', 'plan_edit', 'plan_redo', 'plan_reject',
                            'approve', 'edit', 'reject', 'note')),
    text                TEXT,
    -- How the feedback was sorted: fix this episode / lasting instruction / story change.
    classification      TEXT CHECK (classification IS NULL OR json_valid(classification)),
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Lasting instructions ("slow down the romance"). Included in every future
-- episode while active.
CREATE TABLE IF NOT EXISTS directives (
    id                INTEGER PRIMARY KEY,
    story_id          INTEGER NOT NULL REFERENCES stories(id),
    text              TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('style', 'pacing', 'character', 'plot', 'other')),
    from_feedback_id  INTEGER REFERENCES feedback(id),
    from_ep           INTEGER,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- A directive's status is its latest row here. Old rows are never removed.
CREATE TABLE IF NOT EXISTS directive_status (
    id            INTEGER PRIMARY KEY,
    directive_id  INTEGER NOT NULL REFERENCES directives(id),
    status        TEXT NOT NULL CHECK (status IN ('active', 'paused', 'fulfilled', 'superseded')),
    reason        TEXT,
    ep_no         INTEGER,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------- tracing

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    story_id    INTEGER REFERENCES stories(id),
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'ok', 'error', 'stopped')),
    note        TEXT,
    started_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ended_at    TEXT
);

-- Every attempt at every model call, including failed ones.
CREATE TABLE IF NOT EXISTS llm_calls (
    id                 INTEGER PRIMARY KEY,
    run_id             INTEGER REFERENCES runs(id),
    story_id           INTEGER REFERENCES stories(id),
    ep_no              INTEGER,
    node               TEXT NOT NULL,
    attempt            INTEGER NOT NULL,
    model              TEXT NOT NULL,
    temperature        REAL,
    max_tokens         INTEGER,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    latency_ms         INTEGER NOT NULL DEFAULT 0,
    status             TEXT NOT NULL CHECK (status IN ('ok', 'error', 'invalid_output')),
    finish_reason      TEXT,
    error              TEXT,
    messages           TEXT NOT NULL CHECK (json_valid(messages)),
    response           TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS llm_calls_by_episode ON llm_calls (story_id, ep_no);

-- Decisions the system made (retry, revise, send to human, stop on budget).
CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER REFERENCES runs(id),
    story_id    INTEGER REFERENCES stories(id),
    ep_no       INTEGER,
    node        TEXT NOT NULL,
    decision    TEXT NOT NULL,
    detail      TEXT CHECK (detail IS NULL OR json_valid(detail)),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------- guards

-- Human records can never be deleted.
CREATE TRIGGER IF NOT EXISTS feedback_no_delete BEFORE DELETE ON feedback
BEGIN SELECT RAISE(ABORT, 'feedback is append-only'); END;

CREATE TRIGGER IF NOT EXISTS feedback_no_update BEFORE UPDATE ON feedback
BEGIN SELECT RAISE(ABORT, 'feedback is append-only'); END;

CREATE TRIGGER IF NOT EXISTS directives_no_delete BEFORE DELETE ON directives
BEGIN SELECT RAISE(ABORT, 'directives are append-only'); END;

CREATE TRIGGER IF NOT EXISTS directives_no_update BEFORE UPDATE ON directives
BEGIN SELECT RAISE(ABORT, 'directives are append-only; add a directive_status row'); END;

CREATE TRIGGER IF NOT EXISTS directive_status_no_delete BEFORE DELETE ON directive_status
BEGIN SELECT RAISE(ABORT, 'directive_status is append-only'); END;

CREATE TRIGGER IF NOT EXISTS review_settings_no_delete BEFORE DELETE ON review_settings
BEGIN SELECT RAISE(ABORT, 'review_settings is append-only'); END;

-- Episode text is never overwritten or deleted; only its status moves on.
CREATE TRIGGER IF NOT EXISTS episode_text_frozen BEFORE UPDATE OF text, word_count ON episode_versions
BEGIN SELECT RAISE(ABORT, 'episode text is frozen; create a new version'); END;

CREATE TRIGGER IF NOT EXISTS episode_no_delete BEFORE DELETE ON episode_versions
BEGIN SELECT RAISE(ABORT, 'episode versions are never deleted'); END;

-- ---------------------------------------------------------------- live memory

-- Only memory from approved episodes (or from the story rules / the human,
-- where source_version_id is NULL) counts. Everything reads through these views.

CREATE VIEW IF NOT EXISTS live_characters AS
SELECT c.* FROM characters c
LEFT JOIN episode_versions ev ON ev.id = c.source_version_id
WHERE c.source_version_id IS NULL OR ev.status = 'approved';

CREATE VIEW IF NOT EXISTS live_character_states AS
SELECT cs.* FROM character_states cs
LEFT JOIN episode_versions ev ON ev.id = cs.source_version_id
WHERE cs.source_version_id IS NULL OR ev.status = 'approved';

CREATE VIEW IF NOT EXISTS live_relationships AS
SELECT r.* FROM relationships r
LEFT JOIN episode_versions ev ON ev.id = r.source_version_id
WHERE r.source_version_id IS NULL OR ev.status = 'approved';

CREATE VIEW IF NOT EXISTS live_facts AS
SELECT f.* FROM facts f
LEFT JOIN episode_versions ev ON ev.id = f.source_version_id
WHERE (f.source_version_id IS NULL OR ev.status = 'approved')
  AND f.replaced_by IS NULL;

CREATE VIEW IF NOT EXISTS live_threads AS
SELECT t.* FROM threads t
LEFT JOIN episode_versions ev ON ev.id = t.source_version_id
WHERE t.source_version_id IS NULL OR ev.status = 'approved';

CREATE VIEW IF NOT EXISTS live_thread_events AS
SELECT te.* FROM thread_events te
LEFT JOIN episode_versions ev ON ev.id = te.source_version_id
WHERE te.source_version_id IS NULL OR ev.status = 'approved';

CREATE VIEW IF NOT EXISTS live_key_lines AS
SELECT k.* FROM key_lines k
LEFT JOIN episode_versions ev ON ev.id = k.source_version_id
WHERE k.source_version_id IS NULL OR ev.status = 'approved';

-- Latest status per directive.
CREATE VIEW IF NOT EXISTS current_directives AS
SELECT d.*, ds.status, ds.reason AS status_reason
FROM directives d
JOIN directive_status ds ON ds.id = (
    SELECT id FROM directive_status WHERE directive_id = d.id ORDER BY id DESC LIMIT 1
);
