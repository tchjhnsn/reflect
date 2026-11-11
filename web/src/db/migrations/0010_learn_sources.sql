-- Other sources of what a person knows, beside books (Torian, 2026-11-11). An upload keeps a transcript's
-- file and a model's draft of its rows, or what a LinkedIn export holds, sealed; only rows the person confirms become courses, each linked
-- to a subject the person can rate. A credential is a certificate, license or badge, with what the
-- issuer's badge said when it was checked. A vouch is someone else's word: a LinkedIn endorsement of a
-- subject, or a recommendation in their words. None of these set the person's grasp. Everything a person
-- wrote or was given is sealed; nothing here is changed once kept, except a transcript's draft, which the
-- person works through.
CREATE TABLE IF NOT EXISTS learn_upload (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('transcript', 'linkedin')),
  file_name_ct text NOT NULL,
  file_ct text,
  draft_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_upload_user ON learn_upload (user_id);

CREATE TABLE IF NOT EXISTS learn_course (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  upload_id uuid REFERENCES learn_upload(id),
  row_index integer,
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  row_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (upload_id, row_index)
);
CREATE INDEX IF NOT EXISTS learn_course_user ON learn_course (user_id);

CREATE TABLE IF NOT EXISTS learn_credential (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  dedupe_tag text NOT NULL,
  body_ct text NOT NULL,
  checked_ct text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, dedupe_tag)
);
CREATE INDEX IF NOT EXISTS learn_credential_user ON learn_credential (user_id);

CREATE TABLE IF NOT EXISTS learn_credential_subject (
  credential_id uuid NOT NULL REFERENCES learn_credential(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  PRIMARY KEY (credential_id, subject_id)
);

CREATE TABLE IF NOT EXISTS learn_vouch (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('endorsement', 'recommendation')),
  subject_id uuid REFERENCES learn_subject(id),
  source text NOT NULL,
  dedupe_tag text NOT NULL,
  body_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, dedupe_tag)
);
CREATE INDEX IF NOT EXISTS learn_vouch_subject ON learn_vouch (subject_id);
