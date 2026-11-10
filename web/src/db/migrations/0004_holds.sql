-- How a person holds partnership or children: their picks (desire, will, duty, sacred, not for me, unsure)
-- and optional words. Stated only, kept with beliefs and values and apart from profile facts.
-- A new answer closes the open one by setting valid_to once. Ciphertext, per-person key.
CREATE TABLE IF NOT EXISTS person_hold (
  id bigserial PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  subject text NOT NULL CHECK (subject IN ('partnership', 'children')),
  value_ct text NOT NULL,
  valid_from date NOT NULL,
  valid_to date,
  recorded_at timestamptz NOT NULL DEFAULT now(),
  source text NOT NULL CHECK (source = 'stated')
);
CREATE UNIQUE INDEX IF NOT EXISTS person_hold_open ON person_hold (user_id, subject) WHERE valid_to IS NULL;

CREATE OR REPLACE FUNCTION reflect_append_only() RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
  END IF;
  IF TG_TABLE_NAME IN ('person_fact', 'person_relationship', 'person_belief', 'person_hold')
     AND OLD.valid_to IS NULL AND NEW.valid_to IS NOT NULL
     AND (to_jsonb(NEW) - 'valid_to') = (to_jsonb(OLD) - 'valid_to') THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'reflect: % rows are never edited; only valid_to may be set once', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS person_hold_append_only ON person_hold;
CREATE TRIGGER person_hold_append_only BEFORE UPDATE OR DELETE ON person_hold FOR EACH ROW EXECUTE FUNCTION reflect_append_only();
