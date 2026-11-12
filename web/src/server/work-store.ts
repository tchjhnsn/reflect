// The journal's works (migration 0015). A work keeps every changed save as a version; the latest is the text.
// Nothing is updated or deleted. Title and body are sealed with the person's key.
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { cleanWork, isWorkKind, templateById, workFromFile, type WorkKind } from "@reflect/core/works";
import { open, seal } from "./crypto";

export type Work = { id: string; kind: WorkKind; template: string; title: string; body: string; createdAt: string; savedAt: string; versions: number };

const isId = (id: string) => /^[0-9a-f-]{36}$/i.test(id);

/** A new work from a template of the prompt library: its first version is the template's text, under a quote of what it is about when given (a log act). */
export async function createWork(userId: string, templateId: string, about: string | null = null): Promise<string> {
  const t = templateById(templateId);
  if (!t) throw new Error("No such template.");
  const id = randomUUID();
  const c = await db().connect();
  try {
    await c.query("BEGIN");
    await c.query("INSERT INTO work (id, user_id, kind, template) VALUES ($1, $2, $3, $4)", [id, userId, t.kind, t.id]);
    await c.query("INSERT INTO work_version (work_id, seq, title_ct, body_ct) VALUES ($1, 0, $2, $3)", [id, seal(userId, ""), seal(userId, about ? `> ${about.replace(/\s+/g, " ").trim()}\n\n${t.body}` : t.body)]);
    await c.query("COMMIT");
    return id;
  } catch (e) {
    await c.query("ROLLBACK");
    throw e;
  } finally {
    c.release();
  }
}

type Row = { id: string; kind: string; template: string; created_at: Date; seq: number; title_ct: string; body_ct: string; saved_at: Date };
const toWork = (userId: string, r: Row): Work => ({
  id: r.id, kind: isWorkKind(r.kind) ? r.kind : "entry", template: r.template, title: open(userId, r.title_ct), body: open(userId, r.body_ct),
  createdAt: r.created_at.toISOString(), savedAt: r.saved_at.toISOString(), versions: r.seq + 1,
});
const LATEST = `SELECT w.id, w.kind, w.template, w.created_at, v.seq, v.title_ct, v.body_ct, v.created_at AS saved_at
  FROM work w JOIN LATERAL (SELECT * FROM work_version WHERE work_id = w.id ORDER BY seq DESC LIMIT 1) v ON true`;

/** Every work, most recently saved first. */
export async function listWorks(userId: string): Promise<Work[]> {
  const rows = (await db().query(`${LATEST} WHERE w.user_id = $1 ORDER BY v.created_at DESC`, [userId])).rows as Row[];
  return rows.map((r) => toWork(userId, r));
}

export async function getWork(userId: string, id: string): Promise<Work | null> {
  if (!isId(id)) return null;
  const r = (await db().query(`${LATEST} WHERE w.user_id = $1 AND w.id = $2`, [userId, id])).rows[0] as Row | undefined;
  return r ? toWork(userId, r) : null;
}

/**
 * Save a work: a new version when the text changed, nothing when it did not. Two saves racing for the same
 * version number cannot both land (the primary key refuses the second), so the caller saves again.
 */
export async function saveWork(userId: string, id: string, title: unknown, body: unknown): Promise<{ saved: boolean; savedAt: string }> {
  const cur = await getWork(userId, id);
  if (!cur) throw new Error("No such work.");
  const next = cleanWork(title, body);
  if (next.title === cur.title && next.body === cleanWork("", cur.body).body) return { saved: false, savedAt: cur.savedAt };
  const r = await db().query("INSERT INTO work_version (work_id, seq, title_ct, body_ct) VALUES ($1, $2, $3, $4) RETURNING created_at", [id, cur.versions, seal(userId, next.title), seal(userId, next.body)]);
  return { saved: true, savedAt: (r.rows[0].created_at as Date).toISOString() };
}

/** A work from an imported file, kept as a journal entry; its first version is the file's text. */
export async function importWork(userId: string, fileName: string, text: string): Promise<string> {
  if (!text.trim()) throw new Error("The file is empty.");
  const { title, body } = workFromFile(fileName, text);
  const id = randomUUID();
  const c = await db().connect();
  try {
    await c.query("BEGIN");
    await c.query("INSERT INTO work (id, user_id, kind, template) VALUES ($1, $2, 'entry', 'import')", [id, userId]);
    await c.query("INSERT INTO work_version (work_id, seq, title_ct, body_ct) VALUES ($1, 0, $2, $3)", [id, seal(userId, title), seal(userId, body)]);
    await c.query("COMMIT");
    return id;
  } catch (e) {
    await c.query("ROLLBACK");
    throw e;
  } finally {
    c.release();
  }
}
