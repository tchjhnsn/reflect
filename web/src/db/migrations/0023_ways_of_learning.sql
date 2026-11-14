-- Ways of learning (Torian, 2026-11-14 00:25Z, "I'll take your recommendation on both"): shelves arranged by
-- Aristotle's ways of learning (teaching, dialectic, imitation, practice), with the medium a property of each item.
-- A book may be moved to another way by the person; until then its way follows its Thema category. Lectures,
-- debates, discussions, podcasts, films, music, art and games are learn items: the medium, the way, a status, the
-- title and source sealed together, and what it taught the person sealed apart. Every statement here can run again.
ALTER TABLE learn_book ADD COLUMN IF NOT EXISTS way text CHECK (way IN ('teaching', 'dialectic', 'imitation', 'practice'));
CREATE TABLE IF NOT EXISTS learn_item (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  medium text NOT NULL,
  way text CHECK (way IN ('teaching', 'dialectic', 'imitation', 'practice')),
  status text NOT NULL DEFAULT 'done' CHECK (status IN ('want', 'doing', 'done')),
  body_ct text NOT NULL,
  lesson_ct text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_item_user ON learn_item (user_id, created_at);
