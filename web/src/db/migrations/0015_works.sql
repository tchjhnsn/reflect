-- The journal as a library of works (Torian, 2026-11-12 18:48Z: "This essentially needs to become a writing
-- editor"). A work is one piece the person writes; each save that changes it adds a version, and a version is
-- never changed or deleted, so nothing written is lost. The current text is the latest version. Title and
-- body are sealed with the person's key. Every statement here can run again.
CREATE TABLE IF NOT EXISTS work (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('entry', 'essay', 'letter', 'plan', 'examination', 'exercise', 'decision', 'memoir')),
  template text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS work_user ON work (user_id, created_at);
CREATE TABLE IF NOT EXISTS work_version (
  work_id uuid NOT NULL REFERENCES work(id),
  seq int NOT NULL,
  title_ct text NOT NULL,
  body_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (work_id, seq)
);
CREATE OR REPLACE FUNCTION reflect_never_changed() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS work_append_only ON work;
CREATE TRIGGER work_append_only BEFORE UPDATE OR DELETE ON work FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
DROP TRIGGER IF EXISTS work_version_append_only ON work_version;
CREATE TRIGGER work_version_append_only BEFORE UPDATE OR DELETE ON work_version FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
