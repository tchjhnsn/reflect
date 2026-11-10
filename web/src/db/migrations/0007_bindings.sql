-- What each element of an episode was attached to: a time, a place, a person or a phrase, or a set of
-- them. A cue is kept once per person (found again by a keyed hash, stored as ciphertext). A binding is
-- one element, in one entry, with a weight, attached to one or more cues. Bindings kept together are one
-- reading of the entry; the first is the entry as felt then, and later ones never replace it. Every row
-- is written once.
CREATE TABLE IF NOT EXISTS cue (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('time', 'place', 'person', 'phrase')),
  value_ct text NOT NULL,
  value_tag text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, kind, value_tag)
);

CREATE TABLE IF NOT EXISTS binding (
  id uuid PRIMARY KEY,
  entry_id uuid NOT NULL REFERENCES journal_entry(id),
  user_id text NOT NULL REFERENCES "user"(id),
  element text NOT NULL,
  weight real NOT NULL CHECK (weight >= 0 AND weight <= 1),
  read_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS binding_entry ON binding (entry_id, read_at);

CREATE TABLE IF NOT EXISTS binding_cue (
  binding_id uuid NOT NULL REFERENCES binding(id),
  cue_id uuid NOT NULL REFERENCES cue(id),
  PRIMARY KEY (binding_id, cue_id)
);
CREATE INDEX IF NOT EXISTS binding_cue_cue ON binding_cue (cue_id);

DROP TRIGGER IF EXISTS cue_once ON cue;
CREATE TRIGGER cue_once BEFORE UPDATE OR DELETE ON cue FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS binding_once ON binding;
CREATE TRIGGER binding_once BEFORE UPDATE OR DELETE ON binding FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS binding_cue_once ON binding_cue;
CREATE TRIGGER binding_cue_once BEFORE UPDATE OR DELETE ON binding_cue FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
