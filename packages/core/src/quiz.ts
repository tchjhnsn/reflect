// Quizzing yourself on a lecture (Torian, 2026-11-14), the first steps before dialogue and examination: recall
// what it said (Aristotle, On Memory 452a: recollecting grows with exercise), find what you do not know (Meno
// 84a-c), and give the cause (Meno 98a; Posterior Analytics I.2). A model writes the questions from the lecture's
// transcript, each with the lecture's own answer and the passage it comes from; the person answers, reads the
// lecture's answer and says whether they knew it. Nothing here is graded or goes on any record: examiners are people.

import type { Passage } from "./courses";

export type QuizKind = "recall" | "cause";
export const QUIZ_KINDS: Record<QuizKind, string> = { recall: "What it said", cause: "Why" };
export type Question = { kind: QuizKind; question: string; answer: string; passage: number | null };

export const QUIZ = { questions: 10, question: 400, answer: 1200, answerByPerson: 4000, transcript: 60000 } as const;

/** The transcript as numbered paragraphs, cut at a length a model can read whole. */
export function numberedTranscript(transcript: readonly Passage[], max: number = QUIZ.transcript): string {
  const lines: string[] = [];
  let size = 0;
  for (const [i, p] of transcript.entries()) {
    const line = `[${i}] ${p.heading ? "## " : p.who ? `${p.who}: ` : ""}${p.text.replace(/\s+/g, " ").trim()}`;
    if (size + line.length > max) break;
    lines.push(line);
    size += line.length + 1;
  }
  return lines.join("\n");
}

/** What the model is asked: questions on what the lecture said and on the causes it gives, answered from it alone. */
export function quizPrompt(title: string, transcript: readonly Passage[]): string {
  return [
    `Below is the transcript of a lecture, "${title}", in numbered paragraphs.`,
    "Write 8 questions a student can use to quiz themselves on it: 5 that ask what the lecture said (kind \"recall\") and 3 that ask why, the causes or reasons the lecturer gives (kind \"cause\").",
    "Answer each from the transcript alone, as the lecturer gives it, in one to three sentences, and give the number of the paragraph where the answer is.",
    "Ask about what matters most in the lecture, not about asides, jokes or course business. Do not use outside knowledge.",
    'Reply with JSON only: {"questions":[{"kind":"recall","question":"...","answer":"...","passage":12}]}',
    "",
    numberedTranscript(transcript),
  ].join("\n");
}

const text = (v: unknown, max: number) => (typeof v === "string" ? v.replace(/\s+/g, " ").trim().slice(0, max) : "");

/** The questions in a model's reply; anything malformed is left out, and a passage outside the transcript becomes null. */
export function parseQuiz(reply: string, passages: number): Question[] {
  const start = reply.indexOf("{");
  const end = reply.lastIndexOf("}");
  if (start < 0 || end <= start) return [];
  let data: unknown;
  try {
    data = JSON.parse(reply.slice(start, end + 1));
  } catch {
    return [];
  }
  const list = (data as { questions?: unknown }).questions;
  if (!Array.isArray(list)) return [];
  const out: Question[] = [];
  for (const q of list) {
    if (!q || typeof q !== "object") continue;
    const r = q as Record<string, unknown>;
    const question = text(r.question, QUIZ.question);
    const answer = text(r.answer, QUIZ.answer);
    if (!question || !answer) continue;
    const n = typeof r.passage === "number" ? r.passage : typeof r.passage === "string" ? Number(r.passage) : NaN;
    out.push({ kind: r.kind === "cause" ? "cause" : "recall", question, answer, passage: Number.isInteger(n) && n >= 0 && n < passages ? n : null });
    if (out.length === QUIZ.questions) break;
  }
  return out;
}

/** A question's answer as the person checked it: what they wrote and whether they knew it. */
export type Checked = { seq: number; answer: string; knew: boolean };

/** What the person did not know yet, in the quiz's order. */
export const gaps = (questions: readonly Question[], checked: readonly Checked[]) =>
  questions.map((q, seq) => ({ ...q, seq })).filter((q) => checked.some((c) => c.seq === q.seq && !c.knew));
