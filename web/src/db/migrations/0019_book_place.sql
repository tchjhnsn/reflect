-- Where the person is in a book they are reading in Reflect (Torian, 2026-11-13 10:43Z, "Yes, you should make
-- it"), so the shelf can show their progress and the place follows them to another device. One row per book on
-- their shelves, replaced as they read: it is a place, not a record, and its history is not kept. `at` is how far
-- down the chapter they are, from 0 to 1. Every statement here can run again.
CREATE TABLE IF NOT EXISTS learn_book_place (
  book_id uuid PRIMARY KEY REFERENCES learn_book(id),
  user_id text NOT NULL REFERENCES "user"(id),
  chapter integer NOT NULL CHECK (chapter > 0),
  chapters integer NOT NULL CHECK (chapters >= chapter),
  at real NOT NULL DEFAULT 0 CHECK (at >= 0 AND at <= 1),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_book_place_user ON learn_book_place (user_id);
