-- Learn subjects placed in the domains of a life (Torian, 2026-11-13 03:13Z, "Placing subjects works";
-- 03:19Z "Go ahead"). The person places a subject (Marriage) in a domain (Line) once, and every claim and book
-- under it shows there; Learn stays the source. A new row each time, the latest counts; domain null takes it
-- out of every domain. Append-only. Every statement here can run again.
CREATE TABLE IF NOT EXISTS subject_domain (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  domain text CHECK (domain IN ('needs', 'learn', 'work', 'belonging', 'line', 'household', 'community', 'polis', 'legacy')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS subject_domain_subject ON subject_domain (subject_id, created_at);
CREATE OR REPLACE FUNCTION reflect_never_changed() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS subject_domain_append_only ON subject_domain;
CREATE TRIGGER subject_domain_append_only BEFORE UPDATE OR DELETE ON subject_domain FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
