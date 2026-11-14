import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { placeOf } from "@reflect/core/ways";
import { db } from "@/db/pool";
import { addBook, getBook } from "./learn-store";
import { addInput } from "./ideas-store";
import {
  addItem, addItemNote, addSession, changeItem, getItem, listItems, mergeInputs, removeItem, restoreItem,
  setBookWay, setItemDates, setItemFacts, setTranscript,
} from "./ways-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("ways of learning on Neon", () => {
  afterAll(() => db().end());

  it("keeps items sealed, places a game under rest until it taught something, and moves a book's way", async () => {
    const user = await newUser("Ways Test");
    const game = await addItem(user, { medium: "game", title: "Chess", by: "Anyone", url: "javascript:alert(1)" });
    await addItem(user, { medium: "debate", title: "Lincoln and Douglas", status: "want" });
    let items = await listItems(user);
    const chess = items.find((i) => i.id === game)!;
    expect(chess).toMatchObject({ medium: "game", way: null, status: "done", title: "Chess", by: "Anyone", url: null, lesson: null });
    expect(placeOf(chess)).toBe("leisure");
    const stored = (await db().query("SELECT body_ct FROM learn_item WHERE id = $1", [game])).rows[0].body_ct as string;
    expect(stored).not.toContain("Chess");
    await changeItem(user, game, { lesson: "Patience: wait for the other side's mistake." });
    await expect(changeItem(await newUser("Other"), game, { way: "teaching" })).rejects.toThrow();
    items = await listItems(user);
    expect(placeOf(items.find((i) => i.id === game)!)).toBe("practice");
    await changeItem(user, game, { lesson: "Control the centre." });
    expect((await listItems(user)).find((i) => i.id === game)!.lesson).toBe("Control the centre.");
    expect((await getItem(user, game))!.lessons.map((l) => l.text)).toEqual(["Patience: wait for the other side's mistake.", "Control the centre."]);
    await changeItem(user, game, { way: "dialectic" });
    expect((await listItems(user)).find((i) => i.id === game)!.way).toBe("dialectic");
    await expect(addItem(user, { medium: "tweet", title: "x" })).rejects.toThrow();
    await removeItem(user, game);
    expect((await listItems(user)).map((i) => i.title)).toEqual(["Lincoln and Douglas"]);
    expect((await listItems(user, { removed: true })).map((i) => i.id)).toEqual([game]);
    await restoreItem(user, game);
    expect((await listItems(user)).map((i) => i.id)).toContain(game);

    const book = await addBook(user, { title: "The Iliad", authors: ["Homer"] });
    expect((await getBook(user, book))!.way).toBeNull();
    await setBookWay(user, book, "imitation");
    expect((await getBook(user, book))!.way).toBe("imitation");
    await setBookWay(user, book, null);
    expect((await getBook(user, book))!.way).toBeNull();
  });

  it("keeps dates, sessions, notes, course parts and a transcript, and merges older inputs once", async () => {
    const user = await newUser("Items Test");
    const now = new Date("2026-11-14T12:00:00Z");
    const course = await addItem(user, { medium: "course", title: "Linear Algebra", status: "want", parts: 34 });
    await changeItem(user, course, { status: "doing", day: "2026-11-13" }, now);
    expect((await getItem(user, course))!).toMatchObject({ status: "doing", startedOn: "2026-11-13", finishedOn: null, parts: 34 });
    await setItemFacts(user, course, { part: 50 });
    expect((await getItem(user, course))!.part).toBe(34);
    await expect(setItemDates(user, course, { startedOn: "2026-11-10", finishedOn: "2026-11-02" }, now)).rejects.toThrow();
    await setItemDates(user, course, { startedOn: "2026-11-01", finishedOn: "2026-11-12" }, now);
    expect((await getItem(user, course))!.finishedOn).toBe("2026-11-12");
    await expect(addSession(user, course, { minutes: 2000 }, now)).rejects.toThrow();
    await addSession(user, course, { day: "2026-11-03", minutes: 45, note: "Elimination" }, now);
    await addItemNote(user, course, { at: 125, quote: "A matrix is a map", note: "Remember" });
    await expect(addItemNote(user, course, {})).rejects.toThrow();
    await setTranscript(user, course, "0:00 hello");
    const full = (await getItem(user, course))!;
    expect(full.sessions).toEqual([expect.objectContaining({ day: "2026-11-03", minutes: 45, note: "Elimination" })]);
    expect(full.notes).toEqual([expect.objectContaining({ at: 125, quote: "A matrix is a map", note: "Remember" })]);
    expect(full).toMatchObject({ transcript: true, transcriptText: "0:00 hello" });
    const sealed = (await db().query("SELECT note_ct FROM learn_item_session WHERE item_id = $1", [course])).rows[0].note_ct as string;
    expect(sealed).not.toContain("Elimination");

    const input = await addInput(user, { kind: "talk", title: "The Last Lecture", speaker: "Randy Pausch" } as never);
    const merged = (await listItems(user)).find((i) => i.id === input);
    expect(merged).toMatchObject({ medium: "lecture", title: "The Last Lecture", by: "Randy Pausch" });
    expect(await mergeInputs(user)).toBe(0);
  });
});
