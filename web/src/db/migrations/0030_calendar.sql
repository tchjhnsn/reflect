-- The calendar, a tool (Torian, 2026-11-14 17:44Z and 17:52Z): each tile has its own calendar, a tag on the event,
-- and the master calendar shows them all. An event is something the person plans on a day, at a time or all day,
-- once or repeating; its title and note are sealed with the person's key. It may point at a record in a tile (work
-- they want, a friend). A removed event keeps its removal time until it is restored. Every statement can run again.
CREATE TABLE IF NOT EXISTS calendar_event (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  tile text CHECK (tile IN ('learn', 'work', 'home', 'body', 'friends', 'mate', 'kin', 'polis', 'legacy')),
  title_ct text NOT NULL,
  note_ct text,
  day date NOT NULL,
  starts time,
  ends time,
  repeat text NOT NULL DEFAULT 'none' CHECK (repeat IN ('none', 'daily', 'weekly', 'monthly', 'yearly')),
  until date,
  record_id uuid REFERENCES life_record(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  removed_at timestamptz
);
CREATE INDEX IF NOT EXISTS calendar_event_user ON calendar_event (user_id, day);
