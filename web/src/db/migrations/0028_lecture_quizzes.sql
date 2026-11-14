-- Quizzing yourself on a lecture (Torian, 2026-11-14). `learn_quiz` keeps the questions a model wrote from a
-- lecture's transcript, sealed, for one person's kept course and lecture; `learn_quiz_check` keeps each answer the
-- person wrote and whether they knew it, once per question. Only the person sees these; nothing is graded and
-- nothing goes on a record. Both append-only. Every statement here can run again.
CREATE TABLE IF NOT EXISTS learn_quiz (
  id uuid PRIMARY KEY,
  user_id text NOT NULL REFERENCES "user"(id),
  item_id uuid NOT NULL REFERENCES learn_item(id),
  unit text NOT NULL CHECK (length(unit) BETWEEN 1 AND 300),
  questions_ct text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learn_quiz_unit ON learn_quiz (user_id, item_id, unit, created_at);
CREATE TABLE IF NOT EXISTS learn_quiz_check (
  quiz_id uuid NOT NULL REFERENCES learn_quiz(id),
  seq integer NOT NULL CHECK (seq >= 0),
  answer_ct text NOT NULL,
  knew boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (quiz_id, seq)
);
DROP TRIGGER IF EXISTS learn_quiz_append_only ON learn_quiz;
CREATE TRIGGER learn_quiz_append_only BEFORE UPDATE OR DELETE ON learn_quiz FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
DROP TRIGGER IF EXISTS learn_quiz_check_append_only ON learn_quiz_check;
CREATE TRIGGER learn_quiz_check_append_only BEFORE UPDATE OR DELETE ON learn_quiz_check FOR EACH ROW EXECUTE FUNCTION reflect_never_changed();
