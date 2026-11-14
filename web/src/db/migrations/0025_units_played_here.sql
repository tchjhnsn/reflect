-- Every experience happens in Reflect (Torian, 2026-11-14 02:45Z): a course is taken lecture by lecture and a
-- podcast heard episode by episode inside Reflect. Each unit (a lecture's number, an episode's guid) keeps where the
-- person is in it and the day they finished it; a note can name the unit it was taken in. Can run again.
CREATE TABLE IF NOT EXISTS learn_item_unit (
  item_id uuid NOT NULL REFERENCES learn_item(id),
  user_id text NOT NULL REFERENCES "user"(id),
  unit text NOT NULL CHECK (length(unit) BETWEEN 1 AND 300),
  place_seconds integer CHECK (place_seconds >= 0),
  done_on date,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (item_id, unit)
);
ALTER TABLE learn_item_note ADD COLUMN IF NOT EXISTS unit text;
