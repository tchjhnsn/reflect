-- Claims in Learn (Torian, 2026-11-12; docs/design/REFLECT_UI_2026-11-12.md section 6). A claim is
-- something the person can hold true or false, apart from a subject they know, maybe under a subject. Its
-- holders are named with citations. The person's stance is belief, disbelief or suspension of judgment; a
-- new row every time, never changed, the latest counts. Everything the person wrote is sealed per person.
-- Every statement here can run again.
CREATE TABLE IF NOT EXISTS learn_claim (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  subject_id uuid REFERENCES learn_subject(id),
  text_tag text NOT NULL,
  text_ct text NOT NULL,
  holders_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, text_tag)
);
CREATE INDEX IF NOT EXISTS learn_claim_user ON learn_claim (user_id);

CREATE TABLE IF NOT EXISTS learn_claim_stance (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  claim_id uuid NOT NULL REFERENCES learn_claim(id),
  stance text NOT NULL CHECK (stance IN ('belief', 'disbelief', 'suspension')),
  note_ct text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_claim_stance_claim ON learn_claim_stance (claim_id);
