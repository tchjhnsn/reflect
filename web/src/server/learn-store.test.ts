import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { addBook, addSubjects, getBook, goalFor, keepPlace, keepRecall, learnView, rankWant, rateSubject, readingDays, setBookStatus, setFinishedOn, setGoal } from "./learn-store";
import { logActs } from "./log-store";
import { actText } from "@reflect/core/log";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("learning on Neon", () => {
  afterAll(() => db().end());

  it("keeps reading days, a yearly goal, the order of Want to read and the day a book was finished", async () => {
    const user = await newUser("Habits Test");
    const a = await addBook(user, { title: "A", status: "reading" });
    const now = new Date("2026-11-13T15:00:00Z");
    await keepPlace(user, a, { chapter: 1, chapters: 3, at: 0.2 }, "2026-11-13", now);
    await keepPlace(user, a, { chapter: 1, chapters: 3, at: 0.4 }, "2026-11-13", now);
    await keepPlace(user, a, { chapter: 2, chapters: 3, at: 0 }, "1999-01-01", now);
    expect(await readingDays(user, "2026-01-01", "2026-12-31")).toEqual(["2026-11-13"]);
    await setGoal(user, 2026, 24);
    expect(await goalFor(user, 2026)).toBe(24);
    await setGoal(user, 2026, 30);
    expect(await goalFor(user, 2026)).toBe(30);
    await setGoal(user, 2026, null);
    expect(await goalFor(user, 2026)).toBeNull();
    await expect(setGoal(user, 2026, 0)).rejects.toThrow(/1 to 1,000/);
    const w1 = await addBook(user, { title: "W1" });
    const w2 = await addBook(user, { title: "W2" });
    await rankWant(user, [w2, w1]);
    const view = await learnView(user);
    expect([view.books.find((b) => b.id === w2)!.wantRank, view.books.find((b) => b.id === w1)!.wantRank]).toEqual([0, 1]);
    await expect(setFinishedOn(user, a, "2026-05-01")).rejects.toThrow(/only a read book/);
    await setBookStatus(user, a, "read");
    await setFinishedOn(user, a, "2026-05-01");
    expect((await getBook(user, a))!.finishedOn).toBe("2026-05-01");
    await expect(setFinishedOn(user, a, "3000-01-01")).rejects.toThrow(/to today/);
  });

  it("keeps one place per book, replaced as the person reads, only on their own shelves", async () => {
    const user = await newUser("Place Test");
    const other = await newUser("Other Test");
    const id = await addBook(user, { title: "Pride and Prejudice", status: "reading" });
    expect((await getBook(user, id))?.place).toBeNull();
    await keepPlace(user, id, { chapter: 2, chapters: 62, at: 0.4 });
    await keepPlace(user, id, { chapter: 5, chapters: 62, at: 1.7 });
    expect((await getBook(user, id))?.place).toEqual({ chapter: 5, chapters: 62, at: 1 });
    expect((await db().query("SELECT count(*)::int AS n FROM learn_book_place WHERE book_id = $1", [id])).rows[0].n).toBe(1);
    await expect(keepPlace(user, id, { chapter: 63, chapters: 62, at: 0 })).rejects.toThrow(/chapter/);
    await expect(keepPlace(other, id, { chapter: 1, chapters: 62, at: 0 })).rejects.toThrow(/no such book/);
  });

  it("keeps every status change, never edited, and the log shows each one", async () => {
    const user = await newUser("Log Test");
    const id = await addBook(user, { title: "Ulysses", status: "want" });
    await setBookStatus(user, id, "reading");
    await setBookStatus(user, id, "reading");
    await setBookStatus(user, id, "abandoned");
    const rows = (await db().query("SELECT status FROM learn_book_status WHERE book_id = $1 ORDER BY created_at", [id])).rows.map((r) => r.status);
    expect(rows).toEqual(["want", "reading", "abandoned"]);
    await expect(db().query("UPDATE learn_book_status SET status = 'read' WHERE book_id = $1", [id])).rejects.toThrow(/append-only|never edited/);
    expect((await logActs(user)).map(actText)).toEqual(["Put Ulysses on your shelves", "Started reading Ulysses", "Put down Ulysses"]);
  });

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
