-- What a person knows, by their own account (Torian, 2026-11-11): books are the center. A book has a
-- status and the subjects it covers; a subject is a node of the person's knowledge graph. A recall is
-- what the person named of a book without looking; a grasp is their own rating of a subject (teach,
-- explain, gist, forgot), often from a book. Recalls and grasps are never changed: a later one is a
-- new row, and the latest grasp counts. Titles, names, notes and recalls are sealed per person; a
-- subject's name is also tagged with a keyed hash so the same subject is found again.
CREATE TABLE IF NOT EXISTS learn_book (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  title_ct text NOT NULL,
  authors_ct text NOT NULL,
  isbn_ct text,
  catalog_ct text,
  status text NOT NULL CHECK (status IN ('want', 'reading', 'read', 'abandoned')),
  started_on date,
  finished_on date,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_book_user ON learn_book (user_id);

CREATE TABLE IF NOT EXISTS learn_subject (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  name_tag text NOT NULL,
  name_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, name_tag)
);

CREATE TABLE IF NOT EXISTS learn_book_subject (
  book_id uuid NOT NULL REFERENCES learn_book(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  PRIMARY KEY (book_id, subject_id)
);

CREATE TABLE IF NOT EXISTS learn_recall (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  book_id uuid NOT NULL REFERENCES learn_book(id),
  text_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_recall_book ON learn_recall (book_id);

CREATE TABLE IF NOT EXISTS learn_grasp (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  subject_id uuid NOT NULL REFERENCES learn_subject(id),
  book_id uuid REFERENCES learn_book(id),
  grasp text NOT NULL CHECK (grasp IN ('teach', 'explain', 'gist', 'forgot')),
  note_ct text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_grasp_subject ON learn_grasp (subject_id, created_at);
