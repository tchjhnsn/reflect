-- Reflecting on an entry starts with labels: which elements were part of it, and the self-terms in it.
-- The model's guess is kept beside the person's decision, because the person corrects the model, not
-- the reverse. Rows are written once. Quotes and terms are ciphertext, per-person key.
CREATE TABLE IF NOT EXISTS entry_label (
  id bigserial PRIMARY KEY,
  entry_id uuid NOT NULL REFERENCES journal_entry(id),
  user_id text NOT NULL REFERENCES "user"(id),
  element text NOT NULL,
  model_weight real CHECK (model_weight IS NULL OR (model_weight >= 0 AND model_weight <= 1)),
  decision text NOT NULL CHECK (decision IN ('confirmed', 'rejected', 'added', 'absent')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entry_label_entry ON entry_label (entry_id);

CREATE TABLE IF NOT EXISTS entry_self_term (
  id bigserial PRIMARY KEY,
  entry_id uuid NOT NULL REFERENCES journal_entry(id),
  user_id text NOT NULL REFERENCES "user"(id),
  quote_ct text NOT NULL,
  term_ct text NOT NULL,
  whose text NOT NULL CHECK (whose IN ('self', 'other')),
  model_basis text,
  basis text,
  decision text NOT NULL CHECK (decision IN ('confirmed', 'corrected', 'rejected')),
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Whether the person lets their labels improve the reader. The latest row wins; off until they say so.
CREATE TABLE IF NOT EXISTS research_consent (
  id bigserial PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  allowed boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS entry_label_once ON entry_label;
CREATE TRIGGER entry_label_once BEFORE UPDATE OR DELETE ON entry_label FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS entry_self_term_once ON entry_self_term;
CREATE TRIGGER entry_self_term_once BEFORE UPDATE OR DELETE ON entry_self_term FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS research_consent_once ON research_consent;
CREATE TRIGGER research_consent_once BEFORE UPDATE OR DELETE ON research_consent FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
