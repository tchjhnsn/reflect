import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { IsoDate } from "@/core/person";
import { addPart, closeEntry, createEntry, keepAnswer, listEntries, startReflection } from "./journal-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);
const DAY = "2026-11-09" as IsoDate;

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("journal store on Neon", () => {
  afterAll(() => db().end());

  it("keeps an entry's parts in order, encrypted, until it closes", async () => {
    const user = await newUser();
    const id = await createEntry(user, "Rain at the park.", DAY, null);
    await addPart(user, id, "We went anyway.");
    const [entry] = await listEntries(user);
    expect(entry?.parts.map((p) => p.text)).toEqual(["Rain at the park.", "We went anyway."]);
    expect(entry?.localDate).toBe(DAY);
    const raw = await db().query("SELECT text_ct FROM journal_part WHERE entry_id = $1", [id]);
    expect(raw.rows.every((r: { text_ct: string }) => !r.text_ct.includes("park"))).toBe(true);
    await closeEntry(user, id);
    await expect(addPart(user, id, "late")).rejects.toThrow(/closed/);
    expect((await listEntries(user))[0]?.closedAt).not.toBeNull();
  });

  it("refuses edits, deletes and another person's entry", async () => {
    const user = await newUser();
    const other = await newUser();
    const id = await createEntry(user, "x", DAY, null);
    await expect(db().query("UPDATE journal_part SET text_ct = 'y' WHERE entry_id = $1", [id])).rejects.toThrow(/written once/);
    await expect(db().query("DELETE FROM journal_entry WHERE id = $1", [id])).rejects.toThrow(/append-only/);
    await expect(addPart(other, id, "not mine")).rejects.toThrow(/no such entry/);
    expect(await listEntries(other)).toEqual([]);
  });

  it("keeps reflection answers once, with the question from the frame, for the owner only", async () => {
    const user = await newUser();
    const other = await newUser();
    const id = await createEntry(user, "I said yes to the move.", DAY, null);
    await closeEntry(user, id);
    await expect(startReflection(other, id, "decision")).rejects.toThrow(/no such entry/);
    await expect(startReflection(user, id, "astrology")).rejects.toThrow(/no such frame/);
    const r = await startReflection(user, id, "decision");
    await keepAnswer(user, r, 0, "I chose it.");
    await keepAnswer(user, r, 5, "My mother expects it.");
    await expect(keepAnswer(other, r, 1, "not mine")).rejects.toThrow(/no such question/);
    await expect(keepAnswer(user, r, 99, "past the end")).rejects.toThrow(/no such question/);
    await expect(keepAnswer(user, r, 0, "again")).rejects.toThrow();
    const [entry] = await listEntries(user);
    const [refl] = entry?.reflections ?? [];
    expect(refl?.frame).toBe("decision");
    expect(refl?.answers.map((a) => [a.facet, a.stance, a.text])).toEqual([
      ["voluntary_involuntary", "descriptive", "I chose it."],
      ["conscience", "normative", "My mother expects it."],
    ]);
    const raw = await db().query("SELECT text_ct FROM reflection_answer WHERE reflection_id = $1", [r]);
    expect(raw.rows.every((x: { text_ct: string }) => !x.text_ct.includes("mother"))).toBe(true);
    await expect(db().query("DELETE FROM reflection_answer WHERE reflection_id = $1", [r])).rejects.toThrow(/written once/);
  });
});
