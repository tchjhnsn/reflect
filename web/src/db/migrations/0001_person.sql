-- The person's stores. Append-only: a new value closes the old row by setting valid_to once.
-- Text the person wrote is stored as ciphertext (AES-256-GCM, per-person key).
CREATE TABLE IF NOT EXISTS person_fact (
  id bigserial PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  key text NOT NULL,
  value_ct text NOT NULL,
  valid_from date NOT NULL,
  valid_to date,
  recorded_at timestamptz NOT NULL DEFAULT now(),
  source text NOT NULL CHECK (source = 'stated')
);
CREATE UNIQUE INDEX IF NOT EXISTS person_fact_open ON person_fact (user_id, key) WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS person_relationship (
  id bigserial PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('closest_friend')),
  name_ct text NOT NULL,
  valid_from date NOT NULL,
  valid_to date,
  recorded_at timestamptz NOT NULL DEFAULT now(),
  source text NOT NULL CHECK (source = 'stated')
);
CREATE UNIQUE INDEX IF NOT EXISTS person_relationship_open ON person_relationship (user_id, kind) WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS onboarding_done (
  user_id text PRIMARY KEY REFERENCES "user"(id),
  done_at timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION reflect_append_only() RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
  END IF;
  IF TG_TABLE_NAME IN ('person_fact', 'person_relationship')
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
  FOREACH t IN ARRAY ARRAY['person_fact', 'person_relationship', 'onboarding_done'] LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS %I_append_only ON %I', t, t);
    EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reflect_append_only()', t, t);
  END LOOP;
END $$;
