import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { placeOf } from "@reflect/core/ways";
import { db } from "@/db/pool";
import { addBook, getBook } from "./learn-store";
import { addItem, changeItem, listItems, removeItem, setBookWay } from "./ways-store";

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
    await changeItem(await newUser("Other"), game, { way: "teaching" });
    items = await listItems(user);
    expect(placeOf(items.find((i) => i.id === game)!)).toBe("practice");
    await changeItem(user, game, { way: "dialectic" });
    expect((await listItems(user)).find((i) => i.id === game)!.way).toBe("dialectic");
    await expect(addItem(user, { medium: "tweet", title: "x" })).rejects.toThrow();
    await removeItem(user, game);
    expect((await listItems(user)).map((i) => i.title)).toEqual(["Lincoln and Douglas"]);

    const book = await addBook(user, { title: "The Iliad", authors: ["Homer"] });
    expect((await getBook(user, book))!.way).toBeNull();
    await setBookWay(user, book, "imitation");
    expect((await getBook(user, book))!.way).toBe("imitation");
    await setBookWay(user, book, null);
    expect((await getBook(user, book))!.way).toBeNull();
  });
});
