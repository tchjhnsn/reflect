-- The journal. Entries close once; parts are written once and never edited. Text is ciphertext.
CREATE TABLE IF NOT EXISTS journal_entry (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  local_date date NOT NULL,
  prompt_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  closed_at timestamptz
);
CREATE INDEX IF NOT EXISTS journal_entry_user ON journal_entry (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS journal_part (
  entry_id uuid NOT NULL REFERENCES journal_entry(id),
  seq int NOT NULL,
  text_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entry_id, seq)
);

CREATE OR REPLACE FUNCTION reflect_journal_entry_guard() RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'reflect: journal entries are append-only';
  END IF;
  IF OLD.closed_at IS NULL AND NEW.closed_at IS NOT NULL
     AND (to_jsonb(NEW) - 'closed_at') = (to_jsonb(OLD) - 'closed_at') THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'reflect: journal entries are never edited; they close once';
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reflect_journal_part_guard() RETURNS trigger AS $$
BEGIN
  IF TG_OP <> 'INSERT' THEN
    RAISE EXCEPTION 'reflect: journal parts are written once';
  END IF;
  IF EXISTS (SELECT 1 FROM journal_entry WHERE id = NEW.entry_id AND closed_at IS NOT NULL) THEN
    RAISE EXCEPTION 'reflect: this entry is closed';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS journal_entry_guard ON journal_entry;
CREATE TRIGGER journal_entry_guard BEFORE UPDATE OR DELETE ON journal_entry FOR EACH ROW EXECUTE FUNCTION reflect_journal_entry_guard();
DROP TRIGGER IF EXISTS journal_part_guard ON journal_part;
CREATE TRIGGER journal_part_guard BEFORE INSERT OR UPDATE OR DELETE ON journal_part FOR EACH ROW EXECUTE FUNCTION reflect_journal_part_guard();
