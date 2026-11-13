-- Shelves and habits (Torian, 2026-11-13 18:07Z, "Implement them all"): the order of Want to read as the person
-- drags it, a reading goal for a year if they set one, and the days they read in Reflect (a day is kept when their
-- place in a book is kept, in their own time zone's date). A goal is replaced when changed; a reading day is kept
-- once. Every statement here can run again.
ALTER TABLE learn_book ADD COLUMN IF NOT EXISTS want_rank integer;
CREATE TABLE IF NOT EXISTS learn_goal (
  user_id text NOT NULL REFERENCES "user"(id),
  year integer NOT NULL CHECK (year BETWEEN 1900 AND 3000),
  books integer NOT NULL CHECK (books BETWEEN 1 AND 1000),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, year)
);
CREATE TABLE IF NOT EXISTS learn_reading_day (
  user_id text NOT NULL REFERENCES "user"(id),
  day date NOT NULL,
  PRIMARY KEY (user_id, day)
);
