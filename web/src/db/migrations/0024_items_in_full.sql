-- Learn items in full (Torian, 2026-11-14 01:09Z, "All"): dates, a removal that can be undone, a place in a video,
-- parts of a course, whether the person took part in a discussion, a pasted transcript; lessons kept as dated
-- entries (never edited, as excerpts are), subjects, practice sessions, notes at a moment in a video or recording,
-- claims argued in a debate, and published lessons. Every statement here can run again.
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS started_on date;
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS finished_on date;
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS removed_at timestamptz;
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS took_part boolean NOT NULL DEFAULT false;
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS part integer CHECK (part BETWEEN 0 AND 10000);
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS parts integer CHECK (parts BETWEEN 1 AND 10000);
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS place_seconds integer CHECK (place_seconds >= 0);
ALTER TABLE learn_item ADD COLUMN IF NOT EXISTS transcript_ct text;

CREATE TABLE IF NOT EXISTS learn_item_lesson (
  id uuid PRIMARY KEY,
  item_id uuid NOT NULL REFERENCES learn_item(id),
  user_id text NOT NULL REFERENCES "user"(id),
  text_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_item_lesson_item ON learn_item_lesson (item_id, created_at);
-- A lesson written before this migration becomes the item's first dated lesson (same id as its item, so once).
INSERT INTO learn_item_lesson (id, item_id, user_id, text_ct, created_at)
  SELECT id, id, user_id, lesson_ct, updated_at FROM learn_item WHERE lesson_ct IS NOT NULL
  ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS learn_item_subject (
  item_id uuid NOT NULL REFERENCES learn_item(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  PRIMARY KEY (item_id, subject_id)
);

CREATE TABLE IF NOT EXISTS learn_item_session (
  id uuid PRIMARY KEY,
  item_id uuid NOT NULL REFERENCES learn_item(id),
  user_id text NOT NULL REFERENCES "user"(id),
  day date NOT NULL,
  minutes integer CHECK (minutes BETWEEN 1 AND 1440),
  note_ct text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_item_session_item ON learn_item_session (item_id, day);

CREATE TABLE IF NOT EXISTS learn_item_note (
  id uuid PRIMARY KEY,
  item_id uuid NOT NULL REFERENCES learn_item(id),
  user_id text NOT NULL REFERENCES "user"(id),
  at_seconds integer CHECK (at_seconds >= 0),
  quote_ct text,
  note_ct text,
  removed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_item_note_item ON learn_item_note (item_id, at_seconds);

CREATE TABLE IF NOT EXISTS learn_item_claim (
  item_id uuid NOT NULL REFERENCES learn_item(id),
  claim_id uuid NOT NULL REFERENCES learn_claim(id),
  PRIMARY KEY (item_id, claim_id)
);

ALTER TABLE published_quote ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'quote' CHECK (kind IN ('quote', 'lesson'));
