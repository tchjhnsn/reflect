import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { addExcerpt, allExcerpts, cleanAnchor, excerptsOf, removeExcerpt, reviseExcerpt } from "./excerpt-store";
import { publishedBy, publishedQuote, publishQuote, unpublishQuote } from "./published-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);
const anchor = { source: "gutenberg:1342", chapter: 1, chapterTitle: "Chapter I", from: [0, 0] as [number, number], to: [0, 54] as [number, number] };

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe("anchors", () => {
  it("keeps a highlight's place only when it is whole and runs forward", () => {
    expect(cleanAnchor(anchor)).toEqual(anchor);
    expect(cleanAnchor({ ...anchor, to: [0, 0] })).toBeNull();
    expect(cleanAnchor({ ...anchor, source: "archive:x" })).toBeNull();
    expect(cleanAnchor({ ...anchor, chapter: 0 })).toBeNull();
    expect(cleanAnchor({ ...anchor, from: [0, -1] })).toBeNull();
  });
});

describe.skipIf(!live)("excerpts on Neon", () => {
  afterAll(() => db().end());

  it("keeps highlights and typed excerpts per book, sealed, never edited, and removable only by their owner", async () => {
    const user = await newUser("Excerpt Test");
    const other = await newUser("Other Excerpt Test");
    const h = await addExcerpt(user, { work: "OL66554W", quote: "  It is a truth universally \n  acknowledged ", anchor });
    await addExcerpt(user, { work: "OL66554W", quote: "A typed passage.", place: "p. 12", note: "Why it matters" });
    await addExcerpt(user, { work: "OL1W", quote: "Another book." });
    const got = await excerptsOf(user, "OL66554W");
    expect(got.map((e) => [e.quote, e.place, e.note])).toEqual([["It is a truth universally acknowledged", null, null], ["A typed passage.", "p. 12", "Why it matters"]]);
    expect(got[0]!.anchor).toEqual(anchor);
    expect(await excerptsOf(other, "OL66554W")).toEqual([]);
    const row = (await db().query("SELECT quote_ct, work_tag FROM learn_excerpt WHERE id = $1", [h])).rows[0];
    expect(row.quote_ct).not.toContain("truth");
    expect(row.work_tag).not.toContain("OL66554W");
    await expect(db().query("UPDATE learn_excerpt SET note_ct = NULL WHERE id = $1", [h])).rejects.toThrow(/never edited/);
    await expect(removeExcerpt(other, h)).rejects.toThrow(/no such excerpt/);
    await removeExcerpt(user, h);
    expect((await excerptsOf(user, "OL66554W")).length).toBe(1);
    await expect(addExcerpt(user, { work: "OL66554W", quote: "x", anchor: { ...anchor, to: [0, 0] } })).rejects.toThrow(/not valid/);
    await expect(addExcerpt(user, { work: "nope", quote: "x" })).rejects.toThrow(/no such book/);
  });

  it("changes a highlight's colour and note as a new row with the same date, and lists every book's excerpts", async () => {
    const user = await newUser("Revise Test");
    const id = await addExcerpt(user, { work: "OL66554W", quote: "Coloured.", anchor, color: "green" });
    const before = (await excerptsOf(user, "OL66554W"))[0]!;
    expect(before.color).toBe("green");
    const next = await reviseExcerpt(user, id, { note: "  My note " });
    expect(next).not.toBe(id);
    const after = (await excerptsOf(user, "OL66554W"))[0]!;
    expect([after.id, after.color, after.note, after.at, after.anchor]).toEqual([next, "green", "My note", before.at, anchor]);
    const third = await reviseExcerpt(user, next, { color: "yellow", note: "" });
    expect((await excerptsOf(user, "OL66554W")).map((e) => [e.id, e.color, e.note])).toEqual([[third, "yellow", null]]);
    await expect(reviseExcerpt(user, id, { note: "x" })).rejects.toThrow(/no such excerpt/);
    await addExcerpt(user, { work: "OL1W", quote: "Other book." });
    expect((await allExcerpts(user)).map((e) => e.work).sort()).toEqual(["OL1W", "OL66554W"]);
  });

  it("publishes a quote only on request and takes it down", async () => {
    const user = await newUser("Publish Test");
    const other = await newUser("Other Publish Test");
    const id = await publishQuote(user, { quote: " It is a  truth ", author: "Jane Austen", title: "Pride and Prejudice", place: "Chapter I", work: "OL66554W" });
    expect(await publishedQuote(id)).toMatchObject({ quote: "It is a truth", author: "Jane Austen", place: "Chapter I", work: "OL66554W" });
    expect((await publishedBy(user)).map((q) => q.id)).toEqual([id]);
    await expect(unpublishQuote(other, id)).rejects.toThrow(/no such quote/);
    await unpublishQuote(user, id);
    expect(await publishedQuote(id)).toBeNull();
    expect(await publishedQuote("../../x")).toBeNull();
  });
});
