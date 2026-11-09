import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { Entry } from "@/core/journal";
import type { IsoDate } from "@/core/person";
import { open, seal } from "./crypto";

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
  return entries.map((e) => ({
    id: e.id,
    localDate: iso(e.local_date),
    createdAt: e.created_at.toISOString(),
    promptId: e.prompt_id,
    closedAt: e.closed_at ? e.closed_at.toISOString() : null,
    parts: byEntry.get(e.id) ?? [],
  }));
}
