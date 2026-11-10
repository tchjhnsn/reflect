-- Prompt pack choices: an append-only log; the latest row per pack is the person's choice.
CREATE TABLE IF NOT EXISTS pack_choice (
  id bigserial PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  pack_id text NOT NULL,
  chosen boolean NOT NULL,
  recorded_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pack_choice_user ON pack_choice (user_id, pack_id, id DESC);

-- Beliefs (what the person holds true: descriptive) and values (what they hold matters: normative).
-- Kept apart from profile facts. Stated by the person only; never inferred. Letting one go sets valid_to once.
CREATE TABLE IF NOT EXISTS person_belief (
  id bigserial PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('belief', 'value')),
  text_ct text NOT NULL,
  valid_from date NOT NULL,
  valid_to date,
  recorded_at timestamptz NOT NULL DEFAULT now(),
  source text NOT NULL CHECK (source = 'stated')
);
CREATE INDEX IF NOT EXISTS person_belief_user ON person_belief (user_id, kind);

CREATE OR REPLACE FUNCTION reflect_append_only() RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
  END IF;
  IF TG_TABLE_NAME IN ('person_fact', 'person_relationship', 'person_belief')
     AND OLD.valid_to IS NULL AND NEW.valid_to IS NOT NULL
     AND (to_jsonb(NEW) - 'valid_to') = (to_jsonb(OLD) - 'valid_to') THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'reflect: % rows are never edited; only valid_to may be set once', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['pack_choice', 'person_belief'] LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS %I_append_only ON %I', t, t);
    EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reflect_append_only()', t, t);
  END LOOP;
END $$;
