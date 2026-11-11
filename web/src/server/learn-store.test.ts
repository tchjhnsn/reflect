import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { addBook, addSubjects, getBook, keepRecall, learnView, rateSubject, setBookStatus } from "./learn-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("learning on Neon", () => {
  afterAll(() => db().end());

  it("keeps a book, a recall without looking, and the person's own grasp; read but unrated is not known", async () => {
    const user = await newUser("Reader Test");
    const id = await addBook(user, { title: "Thinking in Systems", authors: ["Donella Meadows"], isbn: "978-1-60358-055-7", status: "want", subjects: ["Systems thinking", "Feedback loops"] });
    await setBookStatus(user, id, "read", new Date("2025-06-01T12:00:00Z"));
    let view = await learnView(user);
    expect(view.known).toEqual([]);
    expect(view.books[0]).toMatchObject({ title: "Thinking in Systems", status: "read", finishedOn: "2025-06-01", startedOn: "2025-06-01", isbn: "9781603580557", unrated: ["Feedback loops", "Systems thinking"] });

    expect(await keepRecall(user, id, "Stocks and flows, and systems thinking.")).toEqual({ named: ["Systems thinking"], unnamed: ["Feedback loops"] });
    await rateSubject(user, "systems thinking", "explain", { bookId: id });
    await rateSubject(user, "Feedback loops", "forgot", { bookId: id, note: "only the bathtub" });
    view = await learnView(user);
    expect(view.known.map((s) => [s.name, s.grasp?.grasp])).toEqual([["Systems thinking", "explain"]]);
    expect(view.books[0]!.unrated).toEqual([]);
    await rateSubject(user, "Feedback loops", "gist");
    expect((await learnView(user)).known.map((s) => s.name)).toEqual(["Feedback loops", "Systems thinking"]);

    await addSubjects(user, id, ["Leverage points"]);
    const book = await getBook(user, id);
    expect(book!.subjects.map((s) => s.name)).toEqual(["Feedback loops", "Leverage points", "Systems thinking"]);
    expect(book!.recalls.map((r) => r.text)).toEqual(["Stocks and flows, and systems thinking."]);
    const stored = (await db().query("SELECT title_ct FROM learn_book WHERE id = $1", [id])).rows[0].title_ct as string;
    expect(stored).not.toContain("Thinking");

    await expect(rateSubject(user, "x", "knows")).rejects.toThrow(/grasp is one of/);
    const other = await newUser("Other Test");
    await expect(keepRecall(other, id, "mine now")).rejects.toThrow(/no such book/);
    expect(await getBook(other, id)).toBeNull();
  });
});
