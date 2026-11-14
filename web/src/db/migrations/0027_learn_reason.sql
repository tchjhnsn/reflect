-- Why a person is learning a course (Torian, 2026-11-14), and spirit in place of anger among the origins.
-- `learn_reason` ties a course to the journal work where the person wrote what is causing them to learn it and
-- for the sake of what, with the origins they marked and whether the end is the thing itself or a use. A new
-- row each time it changes; the latest counts; append-only. Origins kept as 'anger' stay and read as spirit.
-- Every statement here can run again.
ALTER TABLE claim_origin DROP CONSTRAINT IF EXISTS claim_origin_origins_check;
ALTER TABLE claim_origin ADD CONSTRAINT claim_origin_origins_check CHECK (origins <@ ARRAY['chance', 'nature', 'compulsion', 'habit', 'reasoning', 'spirit', 'anger', 'appetite']::text[]);
CREATE TABLE IF NOT EXISTS learn_reason (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  course_url text NOT NULL,
  work_id uuid NOT NULL REFERENCES work(id),
  origins text[] NOT NULL CHECK (origins <@ ARRAY['chance', 'nature', 'compulsion', 'habit', 'reasoning', 'spirit', 'appetite']::text[]),
  end_kind text CHECK (end_kind IN ('own-sake', 'use')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_reason_course ON learn_reason (user_id, course_url, created_at);
DROP TRIGGER IF EXISTS learn_reason_append_only ON learn_reason;
CREATE TRIGGER learn_reason_append_only BEFORE UPDATE OR DELETE ON learn_reason FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
