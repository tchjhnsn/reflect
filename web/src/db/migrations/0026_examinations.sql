-- Putting yourself under examination (Torian, 2026-11-14 04:30Z): no grades and no model as examiner. A person puts
-- themselves under examination on a course, saying what they will answer for; people they invite examine them on a
-- call in Reflect, and each examiner writes an attestation in their own words, which is kept and never edited. The
-- record is the candidate's: what they defended, their name as they gave it and each attestation are sealed with the
-- candidate's key. Can run again.
CREATE TABLE IF NOT EXISTS examination (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  course_url text NOT NULL CHECK (length(course_url) BETWEEN 1 AND 500),
  course_title text NOT NULL CHECK (length(course_title) BETWEEN 1 AND 300),
  name_ct text NOT NULL,
  claim_ct text NOT NULL,
  invite_token text NOT NULL UNIQUE CHECK (length(invite_token) BETWEEN 16 AND 64),
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
  created_at timestamptz NOT NULL DEFAULT now(),
  closed_at timestamptz
);
CREATE INDEX IF NOT EXISTS examination_user ON examination (user_id, course_url);
CREATE TABLE IF NOT EXISTS examination_examiner (
  examination_id uuid NOT NULL REFERENCES examination(id),
  user_id text NOT NULL REFERENCES "user"(id),
  name_ct text NOT NULL,
  joined_at timestamptz NOT NULL DEFAULT now(),
  attests boolean,
  attestation_ct text,
  attested_at timestamptz,
  PRIMARY KEY (examination_id, user_id),
  CHECK ((attested_at IS NULL) = (attestation_ct IS NULL) AND (attested_at IS NULL) = (attests IS NULL))
);
