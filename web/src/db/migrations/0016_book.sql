-- The book of your life (Torian, 2026-11-12 20:12Z, "Build it as proposed"). A belief is a Learn claim
-- (migration 0012). `claim_domain` places it in a chapter, one of the Ventures of a Lifetime domains, chosen by
-- the person; a new row each time, the latest counts. `claim_source` ties a claim to the passage in one of the
-- person's works it was marked from ("I hold this" in the journal editor); the passage is sealed. Both are
-- append-only. Every statement here can run again.
CREATE TABLE IF NOT EXISTS claim_domain (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  claim_id uuid NOT NULL REFERENCES learn_claim(id),
  domain text NOT NULL CHECK (domain IN ('needs', 'learn', 'work', 'belonging', 'line', 'household', 'community', 'polis', 'legacy')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS claim_domain_claim ON claim_domain (claim_id, created_at);
CREATE TABLE IF NOT EXISTS claim_source (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  claim_id uuid NOT NULL REFERENCES learn_claim(id),
  work_id uuid NOT NULL REFERENCES work(id),
  passage_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS claim_source_claim ON claim_source (claim_id);
CREATE OR REPLACE FUNCTION reflect_never_changed() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS claim_domain_append_only ON claim_domain;
CREATE TRIGGER claim_domain_append_only BEFORE UPDATE OR DELETE ON claim_domain FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
DROP TRIGGER IF EXISTS claim_source_append_only ON claim_source;
CREATE TRIGGER claim_source_append_only BEFORE UPDATE OR DELETE ON claim_source FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
