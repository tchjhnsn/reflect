-- Ideals and practical beliefs, and journals inside the domains (Torian, 2026-11-12 21:28Z, "Build").
-- `claim_kind` marks a belief (a Learn claim) as an ideal or a practical belief; a practical belief names the
-- ideal it serves (another claim) and the circumstances it is for (sealed). `claim_origin` records which of
-- Aristotle's seven causes it comes from (Rhetoric I.10); the person marks one or more. `work_domain` files a
-- work of the journal in a domain, or in none (domain null). Each is a new row each time, the latest counts,
-- and all are append-only. Every statement here can run again.
CREATE TABLE IF NOT EXISTS claim_kind (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  claim_id uuid NOT NULL REFERENCES learn_claim(id),
  kind text NOT NULL CHECK (kind IN ('ideal', 'practical')),
  serves_claim_id uuid REFERENCES learn_claim(id),
  circumstances_ct text,
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (kind = 'ideal' OR (serves_claim_id IS NOT NULL AND circumstances_ct IS NOT NULL)),
  CHECK (serves_claim_id IS NULL OR serves_claim_id <> claim_id)
);
CREATE INDEX IF NOT EXISTS claim_kind_claim ON claim_kind (claim_id, created_at);
CREATE TABLE IF NOT EXISTS claim_origin (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  claim_id uuid NOT NULL REFERENCES learn_claim(id),
  origins text[] NOT NULL CHECK (origins <@ ARRAY['chance', 'nature', 'compulsion', 'habit', 'reasoning', 'anger', 'appetite']::text[]),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS claim_origin_claim ON claim_origin (claim_id, created_at);
CREATE TABLE IF NOT EXISTS work_domain (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  work_id uuid NOT NULL REFERENCES work(id),
  domain text CHECK (domain IN ('needs', 'learn', 'work', 'belonging', 'line', 'household', 'community', 'polis', 'legacy')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS work_domain_work ON work_domain (work_id, created_at);
CREATE OR REPLACE FUNCTION reflect_never_changed() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS claim_kind_append_only ON claim_kind;
CREATE TRIGGER claim_kind_append_only BEFORE UPDATE OR DELETE ON claim_kind FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
DROP TRIGGER IF EXISTS claim_origin_append_only ON claim_origin;
CREATE TRIGGER claim_origin_append_only BEFORE UPDATE OR DELETE ON claim_origin FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
DROP TRIGGER IF EXISTS work_domain_append_only ON work_domain;
CREATE TRIGGER work_domain_append_only BEFORE UPDATE OR DELETE ON work_domain FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
