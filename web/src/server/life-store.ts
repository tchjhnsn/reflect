// The records of the apps behind the home screen's other tiles (migration 0029, `@reflect/core/life`). A record's
// fields are sealed together; each state set and each act kept is a new row, never changed, so the log can read
// them. Records belong to one person and are read only by them.
import { randomUUID } from "node:crypto";
import { cleanBody, isAct, isLifeApp, isState, kindOf, type ActFacts, type Body, type LifeAppId } from "@reflect/core/life";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";
import { tx } from "./learn-store";

export type LifeRecord = { id: string; app: LifeAppId; kind: string; state: string | null; body: Body; createdAt: string; updatedAt: string };
export type LifeAct = ActFacts & { id: string; note: string | null; at: string };
export type LifeStateChange = { record: string; state: string; at: string };

const UUID = /^[0-9a-f-]{36}$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;
const iso = (d: unknown) => new Date(d as string).toISOString();

function row(userId: string, r: Record<string, unknown>): LifeRecord {
  return { id: r.id as string, app: r.app as LifeAppId, kind: r.kind as string, state: (r.state as string | null) ?? null, body: JSON.parse(open(userId, r.body_ct as string)) as Body, createdAt: iso(r.created_at), updatedAt: iso(r.updated_at) };
}

/** Every record the person keeps, in one app or in all of them, oldest first. */
export async function lifeRecords(userId: string, app?: LifeAppId): Promise<LifeRecord[]> {
  const r = await db().query(`SELECT id, app, kind, state, body_ct, created_at, updated_at FROM life_record WHERE user_id = $1 AND removed_at IS NULL${app ? " AND app = $2" : ""} ORDER BY created_at, id`, app ? [userId, app] : [userId]);
  return r.rows.map((x) => row(userId, x));
}

export async function lifeRecord(userId: string, id: string): Promise<LifeRecord | null> {
  if (!UUID.test(id)) return null;
  const r = await db().query("SELECT id, app, kind, state, body_ct, created_at, updated_at FROM life_record WHERE id = $1 AND user_id = $2 AND removed_at IS NULL", [id, userId]);
  return r.rows[0] ? row(userId, r.rows[0]) : null;
}

/** Keeps a new record of a kind, with its first state when the kind has states. A kind kept once is kept once. */
export async function addLifeRecord(userId: string, app: string, kind: string, raw: Record<string, unknown>, state?: string): Promise<string> {
  if (!isLifeApp(app)) throw new Error("No such app.");
  const k = kindOf(app, kind);
  if (!k) throw new Error("No such kind.");
  const body = cleanBody(k, raw);
  if (!k.one && body[k.title] === undefined) throw new Error(`Give it a ${k.fields.find((f) => f.key === k.title)!.label.toLowerCase()}.`);
  const first = isState(k, state) ? state : k.states?.[0]?.id ?? null;
  return tx(async (c) => {
    if (k.one) {
      const had = await c.query("SELECT id FROM life_record WHERE user_id = $1 AND app = $2 AND kind = $3 AND removed_at IS NULL LIMIT 1", [userId, app, kind]);
      if (had.rows[0]) throw new Error("You keep one of these already.");
    }
    const id = randomUUID();
    await c.query("INSERT INTO life_record (id, user_id, app, kind, state, body_ct) VALUES ($1, $2, $3, $4, $5, $6)", [id, userId, app, kind, first, seal(userId, JSON.stringify(body))]);
    if (first) await c.query("INSERT INTO life_record_state (id, record_id, user_id, state) VALUES ($1, $2, $3, $4)", [randomUUID(), id, userId, first]);
    return id;
  });
}

/** Changes a record's fields to what the person sent. */
export async function saveLifeRecord(userId: string, id: string, raw: Record<string, unknown>): Promise<void> {
  const r = await lifeRecord(userId, id);
  if (!r) throw new Error("No such record.");
  const k = kindOf(r.app, r.kind)!;
  const body = cleanBody(k, raw);
  if (!k.one && body[k.title] === undefined) throw new Error(`Give it a ${k.fields.find((f) => f.key === k.title)!.label.toLowerCase()}.`);
  await db().query("UPDATE life_record SET body_ct = $3, updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, seal(userId, JSON.stringify(body))]);
}

/** Sets a record's state, keeping the change. Setting the state it is in keeps nothing. */
export async function setLifeState(userId: string, id: string, state: string): Promise<void> {
  const r = await lifeRecord(userId, id);
  if (!r) throw new Error("No such record.");
  if (!isState(kindOf(r.app, r.kind)!, state)) throw new Error("No such state.");
  if (r.state === state) return;
  await tx(async (c) => {
    await c.query("UPDATE life_record SET state = $3, updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, state]);
    await c.query("INSERT INTO life_record_state (id, record_id, user_id, state) VALUES ($1, $2, $3, $4)", [randomUUID(), id, userId, state]);
  });
}

/** Removes a record from view; it is kept with its removal time so it can be restored. */
export async function removeLifeRecord(userId: string, id: string): Promise<void> {
  if (!UUID.test(id)) throw new Error("No such record.");
  await db().query("UPDATE life_record SET removed_at = now() WHERE id = $1 AND user_id = $2 AND removed_at IS NULL", [id, userId]);
}

export async function restoreLifeRecord(userId: string, id: string): Promise<void> {
  if (!UUID.test(id)) throw new Error("No such record.");
  await db().query("UPDATE life_record SET removed_at = NULL WHERE id = $1 AND user_id = $2", [id, userId]);
}

/** Keeps an act on a record: what was done, on what day, for how long or how much, with a note. */
export async function keepLifeAct(userId: string, id: string, a: { kind: string; day: string; minutes?: unknown; amount?: unknown; note?: unknown }, now = new Date()): Promise<void> {
  const r = await lifeRecord(userId, id);
  if (!r) throw new Error("No such record.");
  const k = kindOf(r.app, r.kind)!;
  if (!isAct(k, a.kind)) throw new Error("No such act.");
  const ak = k.acts!.find((x) => x.id === a.kind)!;
  const latest = new Date(now.getTime() + 36 * 3600 * 1000).toISOString().slice(0, 10);
  if (!DAY.test(a.day) || Number.isNaN(Date.parse(a.day)) || a.day > latest || a.day < "1900-01-01") throw new Error("Give the day it happened.");
  const minutes = ak.minutes && a.minutes !== undefined && a.minutes !== "" ? Math.round(Number(a.minutes)) : null;
  if (minutes !== null && !(minutes >= 1 && minutes <= 1440)) throw new Error("Minutes run from 1 to 1440.");
  const amount = ak.amount && a.amount !== undefined && a.amount !== "" ? Number(String(a.amount).replace(/[$,\s]/g, "")) : null;
  if (amount !== null && !(Number.isFinite(amount) && amount >= 0 && amount < 1e10)) throw new Error("Give an amount in dollars.");
  const note = typeof a.note === "string" && a.note.trim() ? seal(userId, a.note.trim().slice(0, 4000)) : null;
  await db().query("INSERT INTO life_act (id, record_id, user_id, kind, day, minutes, amount, note_ct) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)", [randomUUID(), id, userId, a.kind, a.day, minutes, amount, note]);
}

/** The person's acts, on one record or on all of them, newest first. */
export async function lifeActs(userId: string, recordId?: string): Promise<LifeAct[]> {
  const r = await db().query(
    `SELECT a.id, a.record_id, a.kind, a.day::text AS day, a.minutes, a.amount, a.note_ct, a.created_at FROM life_act a JOIN life_record r ON r.id = a.record_id
     WHERE a.user_id = $1 AND r.removed_at IS NULL${recordId ? " AND a.record_id = $2" : ""} ORDER BY a.day DESC, a.created_at DESC`,
    recordId ? [userId, recordId] : [userId],
  );
  return r.rows.map((x) => ({ id: x.id as string, record: x.record_id as string, kind: x.kind as string, day: x.day as string, minutes: (x.minutes as number | null) ?? null, amount: x.amount === null ? null : Number(x.amount), note: x.note_ct ? open(userId, x.note_ct as string) : null, at: iso(x.created_at) }));
}

/** Every state change the person kept, for the log. */
export async function lifeStateChanges(userId: string): Promise<LifeStateChange[]> {
  const r = await db().query("SELECT s.record_id, s.state, s.created_at FROM life_record_state s JOIN life_record r ON r.id = s.record_id WHERE s.user_id = $1 AND r.removed_at IS NULL", [userId]);
  return r.rows.map((x) => ({ record: x.record_id as string, state: x.state as string, at: iso(x.created_at) }));
}
