import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { IsoDate } from "@reflect/core/person";
import { journalContract } from "@reflect/core/journal-contract";
import { filesJournal } from "@reflect/core/files-journal";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { addPart, closeEntry, exportJournal, importJournal, keepChatMessage, postgresJournal, createEntry, entryText, keepAnswer, listEntries, researchAllowed, setResearchAllowed, startReflection } from "./journal-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);
const DAY = "2026-11-09" as IsoDate;

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("journal store on Neon", () => {
  afterAll(() => db().end());

  journalContract(async () => postgresJournal(await newUser()));

  it("moves a journal from a folder to Postgres and back, keeping times", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-move-"));
    try {
      let t = 0;
      const files = filesJournal(dir, "p-move", { now: () => new Date(Date.UTC(2026, 10, 1, 9, 0, t++)) });
      await mkdir(join(dir, "people/p-move/entries"), { recursive: true });
      await writeFile(join(dir, "people/p-move/entries/2026-10-30-walk.md"), "Walked by the river.\n");
      const id = await files.createEntry("Called Dad.", "2026-11-01" as IsoDate, null);
      await files.addPart(id, "We talked about fishing.");
      await files.closeEntry(id);
      const r = await files.startReflection(id, "feeling");
      await files.keepAnswer(r, 0, "Close to him.");
      const before = await files.exportBundle();

      const user = await newUser();
      await importJournal(user, before);
      await expect(importJournal(user, before)).rejects.toThrow(/already has entries/);
      const middle = await exportJournal(user);
      const hand = middle.entries.find((e) => e.parts[0]?.text === "Walked by the river.");
      expect(hand?.id).toMatch(/^[0-9a-f-]{36}$/);
      const norm = (b: typeof before) => ({
        entries: b.entries.map(({ id: _id, reflections, ...e }) => ({ ...e, reflections: (reflections ?? []).map(({ id: _r, ...x }) => x) })),
      });
      expect(norm(middle)).toEqual(norm(before));

      const back = filesJournal(join(dir, "back"), "p-move");
      await back.importBundle(middle);
      expect(norm(await back.exportBundle())).toEqual(norm(before));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

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

  it("keeps a chat as one entry, message by message, for its owner only", async () => {
    const user = await newUser();
    const chat = randomUUID();
    await keepChatMessage(user, chat, DAY, "You: Find me a book on Aristotle");
    await keepChatMessage(user, chat, DAY, "Reflect: Try the Nicomachean Ethics.");
    const [entry] = await listEntries(user);
    expect(entry?.id).toBe(chat);
    expect(entry?.promptId).toBe("chat");
    expect(entry?.parts.map((p) => p.text)).toEqual(["You: Find me a book on Aristotle", "Reflect: Try the Nicomachean Ethics."]);
    await expect(keepChatMessage(await newUser(), chat, DAY, "You: not mine")).rejects.toThrow();
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

  it("answers for one person only, and research stays off until allowed", async () => {
    const user = await newUser();
    const other = await newUser();
    const id = await createEntry(user, "I keep telling myself I'm the independent one.", DAY, null);
    expect(await entryText(user, id)).toContain("independent");
    expect(await entryText(other, id)).toBeNull();
    expect(await researchAllowed(user)).toBe(false);
    await setResearchAllowed(user, true);
    expect(await researchAllowed(user)).toBe(true);
  });
});
