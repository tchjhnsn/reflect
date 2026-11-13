-- Every change of a book's status, kept (Torian, 2026-11-12 09:44Z), so the log can show each act: putting a
-- book on the shelves, starting it, finishing it, putting it down. learn_book keeps the current status; this
-- table keeps the history. A row is never changed or deleted. `on_day` is the day the person gave (it can be
-- earlier than the row, "I read this in 2019"); `created_at` is when it was kept. Books shelved before this
-- table have no rows; the log falls back to their start and finish days. Every statement here can run again.
CREATE TABLE IF NOT EXISTS learn_book_status (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  book_id uuid NOT NULL REFERENCES learn_book(id),
  status text NOT NULL CHECK (status IN ('want', 'reading', 'read', 'abandoned', 'referenced')),
  on_day date NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_book_status_book ON learn_book_status (book_id, created_at);
CREATE OR REPLACE FUNCTION reflect_never_changed() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'reflect: % is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS learn_book_status_append_only ON learn_book_status;
CREATE TRIGGER learn_book_status_append_only BEFORE UPDATE OR DELETE ON learn_book_status FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
