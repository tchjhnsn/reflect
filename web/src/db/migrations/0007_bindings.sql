-- What each element of an episode was attached to. A concept is anything an entry names: a who or what
-- (entity: a person, place, thing, group or idea), a happening or act (event), how something was (quality),
-- or a when (time). It is kept once per person, found again by a keyed hash of its kind, sort and name, and
-- stored as ciphertext. A binding is one element, in one entry, with a weight, attached to one or more
-- concepts; each attachment keeps the entry's own words for the concept and, when given, its role.
-- Bindings kept together are one reading of the entry; the first is the entry as felt then, and later ones
-- never replace it. Every row is written once.
CREATE TABLE IF NOT EXISTS concept (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('entity', 'event', 'quality', 'time')),
  sort text CHECK (sort IS NULL OR (kind = 'entity' AND sort IN ('person', 'place', 'thing', 'group', 'idea'))),
  name_ct text NOT NULL,
  key_tag text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, key_tag)
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

CREATE TABLE IF NOT EXISTS binding_concept (
  binding_id uuid NOT NULL REFERENCES binding(id),
  concept_id uuid NOT NULL REFERENCES concept(id),
  position int NOT NULL,
  words_ct text,
  role text CHECK (role IS NULL OR role IN ('agent', 'patient', 'place', 'time', 'manner', 'cause', 'goal')),
  PRIMARY KEY (binding_id, position)
);
CREATE INDEX IF NOT EXISTS binding_concept_concept ON binding_concept (concept_id);

DROP TRIGGER IF EXISTS concept_once ON concept;
CREATE TRIGGER concept_once BEFORE UPDATE OR DELETE ON concept FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS binding_once ON binding;
CREATE TRIGGER binding_once BEFORE UPDATE OR DELETE ON binding FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
DROP TRIGGER IF EXISTS binding_concept_once ON binding_concept;
CREATE TRIGGER binding_concept_once BEFORE UPDATE OR DELETE ON binding_concept FOR EACH ROW EXECUTE FUNCTION reflect_written_once();
