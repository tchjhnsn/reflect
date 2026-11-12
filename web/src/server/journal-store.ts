import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { Entry, Reflection } from "@reflect/core/journal";
import { frameById } from "@reflect/core/reflect";
import type { IsoDate } from "@reflect/core/person";
import type { Journal } from "@reflect/core/store";
import { makeBundle, type Bundle } from "@reflect/core/bundle";
import { open, seal } from "./crypto";
import { CHAT_PROMPT } from "@reflect/core/chat-journal";

type EntryRow = { id: string; local_date: Date | string; prompt_id: string | null; created_at: Date; closed_at: Date | null };
type PartRow = { entry_id: string; seq: number; text_ct: string; created_at: Date };

const iso = (d: Date | string): IsoDate => (typeof d === "string" ? d.slice(0, 10) : d.toISOString().slice(0, 10)) as IsoDate;

export async function createEntry(userId: string, text: string, localDate: IsoDate, promptId: string | null): Promise<string> {
  const id = randomUUID();
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    await client.query("INSERT INTO journal_entry (id, user_id, local_date, prompt_id) VALUES ($1, $2, $3, $4)", [id, userId, localDate, promptId]);
    await client.query("INSERT INTO journal_part (entry_id, seq, text_ct) VALUES ($1, 0, $2)", [id, seal(userId, text)]);
    await client.query("COMMIT");
    return id;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

export async function addPart(userId: string, entryId: string, text: string): Promise<void> {
  const r = await db().query(
    `INSERT INTO journal_part (entry_id, seq, text_ct)
     SELECT e.id, COALESCE((SELECT max(seq) + 1 FROM journal_part WHERE entry_id = e.id), 0), $3
     FROM journal_entry e WHERE e.id = $1 AND e.user_id = $2`,
    [entryId, userId, seal(userId, text)],
  );
  if (r.rowCount === 0) throw new Error("no such entry");
}

/**
 * Keep one chat message in the journal. The chat's id is its entry's id: the entry is made, marked as a chat,
 * with the first message, and later messages are added to it. Another person's entry with that id is refused.
 */
export async function keepChatMessage(userId: string, chatId: string, localDate: IsoDate, text: string): Promise<void> {
  await db().query("INSERT INTO journal_entry (id, user_id, local_date, prompt_id) VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO NOTHING", [chatId, userId, localDate, CHAT_PROMPT]);
  await addPart(userId, chatId, text);
}

export async function closeEntry(userId: string, entryId: string): Promise<void> {
  await db().query("UPDATE journal_entry SET closed_at = now() WHERE id = $1 AND user_id = $2 AND closed_at IS NULL", [entryId, userId]);
}

export async function listEntries(userId: string, limit = 200): Promise<Entry[]> {
  const entries = (await db().query(
    "SELECT id, local_date::text AS local_date, prompt_id, created_at, closed_at FROM journal_entry WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
    [userId, limit],
  )).rows as EntryRow[];
  if (entries.length === 0) return [];
  const parts = (await db().query(
    "SELECT entry_id, seq, text_ct, created_at FROM journal_part WHERE entry_id = ANY($1::uuid[]) ORDER BY entry_id, seq",
    [entries.map((e) => e.id)],
  )).rows as PartRow[];
  const byEntry = new Map<string, Entry["parts"][number][]>();
  for (const p of parts) {
    const list = byEntry.get(p.entry_id) ?? [];
    list.push({ seq: p.seq, text: open(userId, p.text_ct), createdAt: p.created_at.toISOString() });
    byEntry.set(p.entry_id, list);
  }
  const refl = await reflectionsFor(userId, entries.map((e) => e.id));
  return entries.map((e) => ({
    id: e.id,
    localDate: iso(e.local_date),
    createdAt: e.created_at.toISOString(),
    promptId: e.prompt_id,
    closedAt: e.closed_at ? e.closed_at.toISOString() : null,
    parts: byEntry.get(e.id) ?? [],
    reflections: refl.get(e.id) ?? [],
  }));
}

type ReflRow = { id: string; entry_id: string; frame: string; created_at: Date };
type AnswerRow = { reflection_id: string; seq: number; facet: string; stance: "descriptive" | "normative"; question: string; text_ct: string };

async function reflectionsFor(userId: string, entryIds: string[]): Promise<Map<string, Reflection[]>> {
  const out = new Map<string, Reflection[]>();
  const rows = (await db().query(
    "SELECT id, entry_id, frame, created_at FROM entry_reflection WHERE user_id = $1 AND entry_id = ANY($2::uuid[]) ORDER BY created_at",
    [userId, entryIds],
  )).rows as ReflRow[];
  if (rows.length === 0) return out;
  const answers = (await db().query(
    "SELECT reflection_id, seq, facet, stance, question, text_ct FROM reflection_answer WHERE reflection_id = ANY($1::uuid[]) ORDER BY reflection_id, seq",
    [rows.map((r) => r.id)],
  )).rows as AnswerRow[];
  const byRefl = new Map<string, Reflection["answers"][number][]>();
  for (const a of answers) {
    const list = byRefl.get(a.reflection_id) ?? [];
    list.push({ seq: a.seq, facet: a.facet, stance: a.stance, question: a.question, text: open(userId, a.text_ct) });
    byRefl.set(a.reflection_id, list);
  }
  for (const r of rows) {
    const list = out.get(r.entry_id) ?? [];
    list.push({ id: r.id, frame: r.frame, createdAt: r.created_at.toISOString(), answers: byRefl.get(r.id) ?? [] });
    out.set(r.entry_id, list);
  }
  return out;
}

/** Start reflecting on one of the person's own entries. */
export async function startReflection(userId: string, entryId: string, frame: string): Promise<string> {
  if (!frameById(frame)) throw new Error("no such frame");
  const id = randomUUID();
  const r = await db().query(
    "INSERT INTO entry_reflection (id, entry_id, user_id, frame) SELECT $1, e.id, $3, $4 FROM journal_entry e WHERE e.id = $2 AND e.user_id = $3",
    [id, entryId, userId, frame],
  );
  if (r.rowCount === 0) throw new Error("no such entry");
  return id;
}

/** Keep one answer. The question comes from the frame by its position, never from the client. */
export async function keepAnswer(userId: string, reflectionId: string, seq: number, text: string): Promise<void> {
  const row = (await db().query("SELECT frame FROM entry_reflection WHERE id = $1 AND user_id = $2", [reflectionId, userId])).rows[0] as { frame: string } | undefined;
  const q = row ? frameById(row.frame)?.questions[seq] : undefined;
  if (!q) throw new Error("no such question");
  await db().query(
    "INSERT INTO reflection_answer (reflection_id, seq, facet, stance, question, text_ct) VALUES ($1, $2, $3, $4, $5, $6)",
    [reflectionId, seq, q.facet, q.stance, q.text, seal(userId, text)],
  );
}

/** The person's own words for an entry, joined, for the reader. */
export async function entryText(userId: string, entryId: string): Promise<string | null> {
  const r = await db().query(
    "SELECT p.text_ct FROM journal_part p JOIN journal_entry e ON e.id = p.entry_id WHERE e.id = $1 AND e.user_id = $2 ORDER BY p.seq",
    [entryId, userId],
  );
  if (r.rowCount === 0) return null;
  return (r.rows as { text_ct: string }[]).map((x) => open(userId, x.text_ct)).join("\n\n");
}

export async function researchAllowed(userId: string): Promise<boolean> {
  const r = await db().query("SELECT allowed FROM research_consent WHERE user_id = $1 ORDER BY id DESC LIMIT 1", [userId]);
  return (r.rows[0] as { allowed: boolean } | undefined)?.allowed ?? false;
}

export async function setResearchAllowed(userId: string, allowed: boolean): Promise<void> {
  await db().query("INSERT INTO research_consent (user_id, allowed) VALUES ($1, $2)", [userId, allowed]);
}

/** The Postgres host: one person's journal on Neon, as the store interface the folder host also implements. */
export function postgresJournal(userId: string): Journal {
  return {
    createEntry: (text, localDate, promptId) => createEntry(userId, text, localDate, promptId),
    addPart: (entryId, text) => addPart(userId, entryId, text),
    closeEntry: (entryId) => closeEntry(userId, entryId),
    listEntries: (limit) => listEntries(userId, limit),
    entryText: (entryId) => entryText(userId, entryId),
    startReflection: (entryId, frame) => startReflection(userId, entryId, frame),
    keepAnswer: (reflectionId, seq, text) => keepAnswer(userId, reflectionId, seq, text),
  };
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** The person's whole journal as a bundle, to move it to another host. */
export async function exportJournal(userId: string): Promise<Bundle> {
  return makeBundle(await listEntries(userId, 100_000), new Date().toISOString());
}

/**
 * Import a bundle into a person's empty journal, keeping its times. An id that is not a
 * uuid (a hand-written entry's file name) gets a new one. One transaction: all or nothing.
 */
export async function importJournal(userId: string, b: Bundle): Promise<void> {
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    const has = await client.query("SELECT 1 FROM journal_entry WHERE user_id = $1 LIMIT 1", [userId]);
    if (has.rowCount) throw new Error("this journal already has entries; import goes only into an empty one");
    const ids = new Map<string, string>();
    for (const e of b.entries) {
      const id = UUID.test(e.id) ? e.id : randomUUID();
      ids.set(e.id, id);
      await client.query("INSERT INTO journal_entry (id, user_id, local_date, prompt_id, created_at) VALUES ($1, $2, $3, $4, $5)", [id, userId, e.localDate, e.promptId, e.createdAt]);
      for (const p of e.parts) {
        await client.query("INSERT INTO journal_part (entry_id, seq, text_ct, created_at) VALUES ($1, $2, $3, $4)", [id, p.seq, seal(userId, p.text), p.createdAt]);
      }
      if (e.closedAt) await client.query("UPDATE journal_entry SET closed_at = $2 WHERE id = $1", [id, e.closedAt]);
      for (const r of e.reflections ?? []) {
        const rid = UUID.test(r.id) ? r.id : randomUUID();
        await client.query("INSERT INTO entry_reflection (id, entry_id, user_id, frame, created_at) VALUES ($1, $2, $3, $4, $5)", [rid, id, userId, r.frame, r.createdAt]);
        for (const a of r.answers) {
          await client.query(
            "INSERT INTO reflection_answer (reflection_id, seq, facet, stance, question, text_ct, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            [rid, a.seq, a.facet, a.stance, a.question, seal(userId, a.text), r.createdAt],
          );
        }
      }
    }
    await client.query("COMMIT");
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}
