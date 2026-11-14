import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { addItem } from "./ways-store";
import { checkAnswer, latestQuiz, newQuiz } from "./quiz-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, 'Q', $2, false, now(), now())`, [id, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("lecture quizzes on Neon", () => {
  afterAll(() => db().end());

  it("keeps a quiz sealed for its person, and each answer once", async () => {
    const [ann, ben] = [await newUser(), await newUser()];
    const item = await addItem(ann, { medium: "course", title: "Death", by: "", url: "https://oyc.yale.edu/philosophy/phil-176", image: "", status: "doing", day: "" });
    const questions = [{ kind: "recall" as const, question: "Who disagreed?", answer: "Epicurus.", passage: 2 }, { kind: "cause" as const, question: "Why is death bad?", answer: "It deprives.", passage: 1 }];
    await expect(newQuiz(ben, item, "1", questions)).rejects.toThrow();
    const quiz = await newQuiz(ann, item, "1", questions);
    expect((await db().query("SELECT questions_ct FROM learn_quiz WHERE id = $1", [quiz.id])).rows[0].questions_ct).not.toContain("Epicurus");
    expect(await checkAnswer(ann, quiz.id, 0, "Epicurus", true)).toBe(true);
    expect(await checkAnswer(ann, quiz.id, 0, "Changed", false)).toBe(false);
    expect(await checkAnswer(ann, quiz.id, 5, "x", true)).toBe(false);
    expect(await checkAnswer(ben, quiz.id, 1, "x", true)).toBe(false);
    expect(await checkAnswer(ann, quiz.id, 1, "I am not sure", false)).toBe(true);
    expect(await latestQuiz(ann, item, "1")).toMatchObject({ id: quiz.id, questions, checked: [{ seq: 0, answer: "Epicurus", knew: true }, { seq: 1, answer: "I am not sure", knew: false }] });
    expect(await latestQuiz(ben, item, "1")).toBeNull();
  });
});
