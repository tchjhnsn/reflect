import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { IsoDate } from "@/core/person";
import { addPart, closeEntry, createEntry, listEntries } from "./journal-store";

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
});
