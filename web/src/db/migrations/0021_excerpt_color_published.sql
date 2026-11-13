-- Highlights in colours, and quotes the person publishes (Torian, 2026-11-13 18:07Z, "Implement them all": highlight
-- colours and notes, share a quote as a cited card). A highlight keeps its colour (yellow when none); changing its
-- colour or note keeps a new row and removes the old, since an excerpt is never edited. A published quote is a copy
-- the person chose to make public, in plain text, with its citation: nothing else of theirs is public, and removing
-- it takes the page down. Every statement here can run again.
ALTER TABLE learn_excerpt ADD COLUMN IF NOT EXISTS color text;
CREATE TABLE IF NOT EXISTS published_quote (
  id text PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  quote text NOT NULL,
  author text NOT NULL DEFAULT '',
  title text NOT NULL,
  place text,
  work text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS published_quote_user ON published_quote (user_id, created_at);
