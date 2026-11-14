// Quizzes a person takes on a lecture of a kept course (migration 0028): the questions sealed with their key, and
// each answer they checked, once. Only they see it.
import { randomUUID } from "node:crypto";
import { QUIZ, type Checked, type Question } from "@reflect/core/quiz";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";

export type Quiz = { id: string; questions: Question[]; checked: Checked[]; createdAt: string };

const isId = (id: string) => /^[0-9a-f-]{36}$/i.test(id);

/** Keeps a new quiz on a lecture of a course the person keeps. */
export async function newQuiz(userId: string, itemId: string, unit: string, questions: readonly Question[]): Promise<Quiz> {
  if (!isId(itemId) || !questions.length) throw new Error("There is no quiz to keep.");
  const own = (await db().query("SELECT 1 FROM learn_item WHERE id = $1 AND user_id = $2", [itemId, userId])).rowCount;
  if (!own) throw new Error("That course is not on your shelves.");
  const id = randomUUID();
  const r = await db().query<{ created_at: Date }>("INSERT INTO learn_quiz (id, user_id, item_id, unit, questions_ct) VALUES ($1, $2, $3, $4, $5) RETURNING created_at", [id, userId, itemId, unit, seal(userId, JSON.stringify(questions))]);
  return { id, questions: [...questions], checked: [], createdAt: r.rows[0]!.created_at.toISOString() };
}

/** The person's latest quiz on a lecture, with what they have checked, or null. */
export async function latestQuiz(userId: string, itemId: string, unit: string): Promise<Quiz | null> {
  if (!isId(itemId)) return null;
  const q = (await db().query<{ id: string; questions_ct: string; created_at: Date }>("SELECT id, questions_ct, created_at FROM learn_quiz WHERE user_id = $1 AND item_id = $2 AND unit = $3 ORDER BY created_at DESC, id LIMIT 1", [userId, itemId, unit])).rows[0];
  if (!q) return null;
  const checks = (await db().query<{ seq: number; answer_ct: string; knew: boolean }>("SELECT seq, answer_ct, knew FROM learn_quiz_check WHERE quiz_id = $1 ORDER BY seq", [q.id])).rows;
  return { id: q.id, questions: JSON.parse(open(userId, q.questions_ct)) as Question[], checked: checks.map((c) => ({ seq: c.seq, answer: open(userId, c.answer_ct), knew: c.knew })), createdAt: q.created_at.toISOString() };
}

/** Keeps the person's answer to one question and whether they knew it; once per question. */
export async function checkAnswer(userId: string, quizId: string, seq: number, answer: unknown, knew: boolean): Promise<boolean> {
  if (!isId(quizId) || !Number.isInteger(seq) || seq < 0) return false;
  const q = (await db().query<{ questions_ct: string }>("SELECT questions_ct FROM learn_quiz WHERE id = $1 AND user_id = $2", [quizId, userId])).rows[0];
  if (!q || seq >= (JSON.parse(open(userId, q.questions_ct)) as Question[]).length) return false;
  const text = typeof answer === "string" ? answer.trim().slice(0, QUIZ.answerByPerson) : "";
  const r = await db().query("INSERT INTO learn_quiz_check (quiz_id, seq, answer_ct, knew) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING", [quizId, seq, seal(userId, text), knew]);
  return (r.rowCount ?? 0) === 1;
}
