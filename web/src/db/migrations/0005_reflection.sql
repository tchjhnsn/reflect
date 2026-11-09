-- Reflecting on a journal entry: the person picks a frame and answers fixed, open questions one at a time.
-- Each answer is written once and never edited. Skipped questions leave no row. The question text is
-- stored as it was asked; the answer is ciphertext, per-person key. No model writes here.
CREATE TABLE IF NOT EXISTS entry_reflection (
  id uuid PRIMARY KEY,
  entry_id uuid NOT NULL REFERENCES journal_entry(id),
  user_id text NOT NULL REFERENCES "user"(id),
  frame text NOT NULL CHECK (frame IN ('feeling', 'decision', 'belief', 'identity')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entry_reflection_entry ON entry_reflection (entry_id);

CREATE TABLE IF NOT EXISTS reflection_answer (
  reflection_id uuid NOT NULL REFERENCES entry_reflection(id),
  seq int NOT NULL,
  facet text NOT NULL,
  stance text NOT NULL CHECK (stance IN ('descriptive', 'normative')),
  question text NOT NULL,
  text_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (reflection_id, seq)
);

CREATE OR REPLACE FUNCTION reflect_written_once() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % rows are written once', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS entry_reflection_once ON entry_reflection;
CREATE TRIGGER entry_reflection_once BEFORE UPDATE OR DELETE ON entry_reflection FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS reflection_answer_once ON reflection_answer;
CREATE TRIGGER reflection_answer_once BEFORE UPDATE OR DELETE ON reflection_answer FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
