-- Items and their pipeline (docs/design/REFLECT_SYSTEM_2026-11-12.md sections 3 and 6): every change to an
-- item is an event, dated, added to the item's history and never edited. The item's state is computed by the
-- core (packages/core/src/items.ts) from its events; nothing here stores it. `body_ct` is the event's fields
-- as JSON, sealed with the person's key, since it holds their words. `hash` chains each item's events:
-- sha256 of the previous hash, the item, the sequence number, the kind and the sealed body. A row is never
-- changed or deleted. Every statement here can run again.
CREATE TABLE IF NOT EXISTS item_event (
  item_id uuid NOT NULL,
  seq int NOT NULL,
  user_id text NOT NULL REFERENCES "user"(id),
  kind text NOT NULL CHECK (kind IN ('stated', 'observed_why', 'reflected', 'examine', 'examine_step', 'examined', 'another_look', 'planned', 'scheduled', 'started', 'done', 'missed')),
  body_ct text NOT NULL,
  hash text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (item_id, seq)
);
CREATE INDEX IF NOT EXISTS item_event_user ON item_event (user_id, created_at);
CREATE OR REPLACE FUNCTION reflect_never_changed() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS item_event_append_only ON item_event;
CREATE TRIGGER item_event_append_only BEFORE UPDATE OR DELETE ON item_event FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
