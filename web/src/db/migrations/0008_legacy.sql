-- A legacy release: what of a person's legacy goes to whom after their death, carried out by the app.
-- The person keeps a plan (trustees who may confirm the death, how many must confirm, and how many days
-- of waiting follow, in which the person or any trustee can stop it) and gifts (each person's part,
-- sealed). Each trustee and each recipient has a link whose token is stored hashed for lookup and
-- sealed so the person can see it again. A new plan revokes the old trustees and gifts, never deletes
-- them. A release opens with the first confirmation; once enough trustees confirm it starts, and after
-- the wait each recipient's link shows their part. A stopped release stays as a record.
CREATE TABLE IF NOT EXISTS legacy_plan (
  user_id text PRIMARY KEY REFERENCES "user"(id),
  confirmations integer NOT NULL CHECK (confirmations >= 1),
  wait_days integer NOT NULL CHECK (wait_days >= 0),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS legacy_trustee (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  name_ct text NOT NULL,
  contact_ct text NOT NULL,
  token_hash text NOT NULL UNIQUE,
  token_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz
);
CREATE INDEX IF NOT EXISTS legacy_trustee_user ON legacy_trustee (user_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS legacy_gift (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  recipient_ct text NOT NULL,
  contact_ct text NOT NULL,
  items_ct text NOT NULL,
  token_hash text NOT NULL UNIQUE,
  token_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz
);
CREATE INDEX IF NOT EXISTS legacy_gift_user ON legacy_gift (user_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS legacy_release (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  opened_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  wait_days integer NOT NULL CHECK (wait_days >= 0),
  stopped_at timestamptz,
  stopped_by text
);
-- At most one open release per person.
CREATE UNIQUE INDEX IF NOT EXISTS legacy_release_open ON legacy_release (user_id) WHERE stopped_at IS NULL;

CREATE TABLE IF NOT EXISTS legacy_confirmation (
  release_id uuid NOT NULL REFERENCES legacy_release(id),
  trustee_id uuid NOT NULL REFERENCES legacy_trustee(id),
  confirmed_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (release_id, trustee_id)
);
