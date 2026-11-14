-- The apps behind the home screen's other tiles (Torian, 2026-11-14 15:45Z): Work, Home, Body, Friends, Mate, Kin,
-- Polis and Legacy (`@reflect/core/life`). A record is one thing the person keeps in an app (a piece of work, a
-- friend, a will), its fields sealed together with the person's key. Each state the person sets is a new row in
-- `life_record_state`, and each act they keep (a workout, a visit, a vote) a row in `life_act`; both append-only,
-- so the log reads them. A removed record keeps its removal time until it is restored. Every statement here can
-- run again.
CREATE TABLE IF NOT EXISTS life_record (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  app text NOT NULL CHECK (app IN ('work', 'home', 'body', 'friends', 'mate', 'kin', 'polis', 'legacy')),
  kind text NOT NULL CHECK (kind ~ '^[a-z][a-z-]{0,39}$'),
  state text CHECK (state ~ '^[a-z][a-z-]{0,39}$'),
  body_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  removed_at timestamptz
);
CREATE INDEX IF NOT EXISTS life_record_user ON life_record (user_id, app, kind);

CREATE TABLE IF NOT EXISTS life_record_state (
  id uuid PRIMARY KEY,
  record_id uuid NOT NULL REFERENCES life_record(id),
  user_id text NOT NULL REFERENCES "user"(id),
  state text NOT NULL CHECK (state ~ '^[a-z][a-z-]{0,39}$'),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS life_record_state_record ON life_record_state (record_id, created_at);
DROP TRIGGER IF EXISTS life_record_state_append_only ON life_record_state;
CREATE TRIGGER life_record_state_append_only BEFORE UPDATE OR DELETE ON life_record_state FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();

CREATE TABLE IF NOT EXISTS life_act (
  id uuid PRIMARY KEY,
  record_id uuid NOT NULL REFERENCES life_record(id),
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind ~ '^[a-z][a-z-]{0,39}$'),
  day date NOT NULL,
  minutes integer CHECK (minutes BETWEEN 1 AND 1440),
  amount numeric(12, 2) CHECK (amount >= 0),
  note_ct text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS life_act_user ON life_act (user_id, day);
DROP TRIGGER IF EXISTS life_act_append_only ON life_act;
CREATE TRIGGER life_act_append_only BEFORE UPDATE OR DELETE ON life_act FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
