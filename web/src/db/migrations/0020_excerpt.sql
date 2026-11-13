-- Excerpts the person keeps from a book (Torian, 2026-11-13 11:12Z, "Build it": raw-text reading with highlights
-- and excerpts). A highlight is an excerpt marked in a text read in Reflect: it keeps where it sits (the text's
-- source, chapter, and the paragraph and character it starts and ends at) beside the quoted words, so it is found
-- again even if the text changes a little. An excerpt the person types from a book without a text keeps the page
-- or place they give instead. The book is found by a keyed hash of its Open Library work id (`work_tag`); every
-- word is sealed. The person can remove an excerpt; an excerpt is never edited. Every statement here can run again.
CREATE TABLE IF NOT EXISTS learn_excerpt (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  work_tag text NOT NULL,
  work_ct text NOT NULL,
  quote_ct text NOT NULL,
  anchor_ct text,
  place_ct text,
  note_ct text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_excerpt_work ON learn_excerpt (user_id, work_tag, created_at);
CREATE OR REPLACE FUNCTION reflect_excerpt_never_edited() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: learn_excerpt is never edited';
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS learn_excerpt_never_edited ON learn_excerpt;
CREATE TRIGGER learn_excerpt_never_edited BEFORE UPDATE ON learn_excerpt FOR EACH ROW EXECUTE FUNCTION reflect_excerpt_never_edited();
