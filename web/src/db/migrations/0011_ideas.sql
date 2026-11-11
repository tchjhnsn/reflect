-- Inputs apart from outputs (Torian, 2026-11-11). Films, talks, conversations and links are inputs beside
-- books: what the person took in. A subject mark is evidence beside a subject that is never a grasp:
-- "shown" when someone saw the person display knowledge of it, "suggested" when someone thinks it relates
-- to them. An idea is the person's own output, kept apart from learn: each line says it again in their
-- words, links an input it came from (or may have), places a recommendation from supports to opposes, or
-- drops one. Lines are never changed; the latest line about a thing decides where it stands. A memory sweep
-- (what the person's own AI says it knows of them) is an upload, kept item by item. Everything the person
-- wrote is sealed per person. Every statement here can run again.
ALTER TABLE learn_book DROP CONSTRAINT IF EXISTS learn_book_status_check;
ALTER TABLE learn_book ADD CONSTRAINT learn_book_status_check CHECK (status IN ('want', 'reading', 'read', 'abandoned', 'referenced'));
ALTER TABLE learn_upload DROP CONSTRAINT IF EXISTS learn_upload_kind_check;
ALTER TABLE learn_upload ADD CONSTRAINT learn_upload_kind_check CHECK (kind IN ('transcript', 'linkedin', 'sweep'));

CREATE TABLE IF NOT EXISTS learn_input (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('film', 'talk', 'conversation', 'link')),
  dedupe_tag text NOT NULL,
  body_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, dedupe_tag)
);
CREATE INDEX IF NOT EXISTS learn_input_user ON learn_input (user_id);

CREATE TABLE IF NOT EXISTS learn_input_subject (
  input_id uuid NOT NULL REFERENCES learn_input(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  PRIMARY KEY (input_id, subject_id)
);

CREATE TABLE IF NOT EXISTS learn_subject_mark (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  kind text NOT NULL CHECK (kind IN ('shown', 'suggested')),
  body_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_subject_mark_subject ON learn_subject_mark (subject_id);

CREATE TABLE IF NOT EXISTS idea (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  name_tag text NOT NULL,
  title_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, name_tag)
);

CREATE TABLE IF NOT EXISTS idea_line (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  idea_id uuid NOT NULL REFERENCES idea(id),
  kind text NOT NULL CHECK (kind IN ('stated', 'from', 'maybe', 'related', 'dropped')),
  body_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idea_line_idea ON idea_line (idea_id, created_at);

CREATE OR REPLACE FUNCTION reflect_lines_only() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only; a later line says what changed', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['idea_line', 'learn_subject_mark'] LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS %I_append_only ON %I', t, t);
    EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reflect_lines_only()', t, t);
  END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS learn_upload_kept (
  upload_id uuid NOT NULL REFERENCES learn_upload(id),
  row_index integer NOT NULL,
  kept_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (upload_id, row_index)
);
