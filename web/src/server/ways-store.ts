import { randomUUID } from "node:crypto";
import type { PoolClient } from "@neondatabase/serverless";
import { ITEM_STATUS, isMedium, isWay, type ItemStatus, type Medium, type Way } from "@reflect/core/ways";
import type { Candidate } from "@reflect/core/item-imports";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";
import { subjectId, tx } from "./learn-store";

/**
 * Learn items (migrations 0023 and 0024): a lecture, course, article, debate, discussion, podcast, film, series,
 * theatre, music, art or game the person learned from, or keeps for rest. The title, maker, link, picture and the
 * book it adapts are sealed together; lessons are dated entries, sealed, never edited. The way is the person's
 * choice, or null to follow the medium. A removed item is kept with its removal time until the person restores it.
 */
export type ItemSubject = { id: string; name: string };
export type LearnItem = {
  id: string; medium: Medium; way: Way | null; status: ItemStatus; title: string; by: string | null; url: string | null; image: string | null;
  /** The Open Library work the item adapts or stages, when the person says so. */ work: string | null;
  /** The latest lesson, which decides whether a game, music or art counts as learning. */ lesson: string | null;
  startedOn: string | null; finishedOn: string | null; tookPart: boolean; part: number | null; parts: number | null; place: number | null;
  subjects: ItemSubject[]; transcript: boolean; at: string;
};
export type ItemLesson = { id: string; text: string; at: string };
export type ItemSession = { id: string; day: string; minutes: number | null; note: string | null };
export type ItemNote = { id: string; at: number | null; quote: string | null; note: string | null; created: string; unit: string | null };
export type ItemUnit = { unit: string; place: number | null; doneOn: string | null };
export type ItemFull = LearnItem & { lessons: ItemLesson[]; sessions: ItemSession[]; notes: ItemNote[]; claimIds: string[]; transcriptText: string | null };
export type ItemInput = { medium: string; way?: string | null; status?: string; title: string; by?: string; url?: string; image?: string; work?: string; lesson?: string; startedOn?: string; finishedOn?: string; tookPart?: boolean; parts?: number; day?: string };

type Body = { title: string; by?: string | null; url?: string | null; image?: string | null; work?: string | null };
const UUID = /^[0-9a-f-]{36}$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;
const isStatus = (s: unknown): s is ItemStatus => typeof s === "string" && (ITEM_STATUS as readonly string[]).includes(s);
const clean = (s: string | undefined | null, max: number) => (s ?? "").trim().slice(0, max) || null;
const goodUrl = (u: string | null, https = false) => (u && (https ? /^https:\/\/\S+$/i : /^https?:\/\/\S+$/i).test(u) ? u : null);
const goodWork = (w: string | null | undefined) => (w && /^OL\d+W$/.test(w) ? w : null);
const dayOf = (d: unknown) => (d ? (d instanceof Date ? d.toISOString().slice(0, 10) : String(d).slice(0, 10)) : null);
const okDay = (d: string | undefined | null, now = new Date()) => (d && DAY.test(d) && d >= "1900-01-01" && d <= new Date(now.getTime() + 36 * 3600 * 1000).toISOString().slice(0, 10) && !Number.isNaN(Date.parse(d)) ? d : null);
/** The person's own date when given and within a day and a half of the server's, else the server's. */
const today = (day?: string, now = new Date()) => (day && DAY.test(day) && Math.abs(Date.parse(`${day}T12:00:00Z`) - now.getTime()) < 36 * 3600 * 1000 ? day : now.toISOString().slice(0, 10));

function bodyOf(i: { title: string; by?: string | undefined | null; url?: string | undefined | null; image?: string | undefined | null; work?: string | undefined | null }): Body {
  const title = clean(i.title, 300);
  if (!title) throw new Error("Give it a title.");
  return { title, by: clean(i.by, 200), url: goodUrl(clean(i.url, 1000)), image: goodUrl(clean(i.image, 1000), true), work: goodWork(i.work) };
}

async function own(c: PoolClient | ReturnType<typeof db>, userId: string, id: string): Promise<{ body_ct: string; status: ItemStatus; started_on: unknown }> {
  if (!UUID.test(id)) throw new Error("no such item");
  const r = await c.query("SELECT body_ct, status, started_on FROM learn_item WHERE id = $1 AND user_id = $2", [id, userId]);
  if (!r.rows[0]) throw new Error("no such item");
  return r.rows[0] as { body_ct: string; status: ItemStatus; started_on: unknown };
}

async function keepDay(c: PoolClient, userId: string, day: string): Promise<void> {
  await c.query("INSERT INTO learn_reading_day (user_id, day) VALUES ($1, $2) ON CONFLICT DO NOTHING", [userId, day]);
}

export async function addItem(userId: string, i: ItemInput, now = new Date()): Promise<string> {
  if (!isMedium(i.medium)) throw new Error("Unknown kind.");
  const body = bodyOf(i);
  const status = isStatus(i.status) ? i.status : "done";
  const id = randomUUID();
  const lesson = clean(i.lesson, 4000);
  const parts = Number.isInteger(i.parts) && i.parts! >= 1 && i.parts! <= 10000 ? i.parts! : null;
  await tx(async (c) => {
    await c.query(
      "INSERT INTO learn_item (id, user_id, medium, way, status, body_ct, started_on, finished_on, took_part, parts) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
      [id, userId, i.medium, isWay(i.way) ? i.way : null, status, seal(userId, JSON.stringify(body)), okDay(i.startedOn, now), status === "done" ? okDay(i.finishedOn, now) : null, Boolean(i.tookPart), parts],
    );
    if (lesson) await c.query("INSERT INTO learn_item_lesson (id, item_id, user_id, text_ct) VALUES ($1, $2, $3, $4)", [randomUUID(), id, userId, seal(userId, lesson)]);
    if (status !== "want" && !i.finishedOn && !i.startedOn && i.day) await keepDay(c, userId, today(i.day, now));
  });
  return id;
}

/** Keeps candidates from an import the person picked, skipping a title they already keep in that medium. */
export async function addItems(userId: string, list: Candidate[]): Promise<number> {
  if (list.length > 500) throw new Error("at most 500 at a time");
  const kept = new Set((await listItems(userId)).map((i) => `${i.medium}|${i.title.toLowerCase()}`));
  let n = 0;
  for (const c of list) {
    if (!isMedium(c.medium) || kept.has(`${c.medium}|${c.title.trim().toLowerCase()}`)) continue;
    await addItem(userId, { medium: c.medium, title: c.title, ...(c.by ? { by: c.by } : {}), ...(c.url ? { url: c.url } : {}), status: c.status, ...(c.on && c.status === "done" ? { finishedOn: c.on } : c.on ? { startedOn: c.on } : {}) });
    kept.add(`${c.medium}|${c.title.trim().toLowerCase()}`);
    n++;
  }
  return n;
}

/** Changes the title, maker, link, picture or the book it adapts; the rest of the body is kept. */
export async function editItem(userId: string, id: string, e: Partial<Body>): Promise<void> {
  await tx(async (c) => {
    const row = await own(c, userId, id);
    const old = JSON.parse(open(userId, row.body_ct)) as Body;
    const body = bodyOf({ ...old, ...e });
    await c.query("UPDATE learn_item SET body_ct = $3, updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, seal(userId, JSON.stringify(body))]);
  });
}

/**
 * Changes the way (null follows the medium) or the status. Starting keeps the day it started; finishing keeps the
 * day it was finished; either keeps a day of learning. A non-empty lesson is added as a new dated lesson.
 */
export async function changeItem(userId: string, id: string, ch: { way?: string | null; status?: string; lesson?: string | null; day?: string }, now = new Date()): Promise<void> {
  await tx(async (c) => {
    const row = await own(c, userId, id);
    const d = today(ch.day, now);
    if (ch.way !== undefined) await c.query("UPDATE learn_item SET way = $3, updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, isWay(ch.way) ? ch.way : null]);
    if (ch.status !== undefined && isStatus(ch.status) && ch.status !== row.status) {
      await c.query(
        `UPDATE learn_item SET status = $3, started_on = CASE WHEN $3 = 'doing' AND started_on IS NULL THEN $4::date ELSE started_on END,
         finished_on = CASE WHEN $3 = 'done' THEN $4::date ELSE NULL END, updated_at = now() WHERE id = $1 AND user_id = $2`,
        [id, userId, ch.status, d],
      );
      if (ch.status !== "want") await keepDay(c, userId, d);
    }
    const lesson = clean(ch.lesson, 4000);
    if (lesson) await c.query("INSERT INTO learn_item_lesson (id, item_id, user_id, text_ct) VALUES ($1, $2, $3, $4)", [randomUUID(), id, userId, seal(userId, lesson)]);
  });
}

/** Sets the days it was started and finished (either may be cleared). */
export async function setItemDates(userId: string, id: string, d: { startedOn?: string | null; finishedOn?: string | null }, now = new Date()): Promise<void> {
  const s = d.startedOn ? okDay(d.startedOn, now) : null;
  const f = d.finishedOn ? okDay(d.finishedOn, now) : null;
  if ((d.startedOn && !s) || (d.finishedOn && !f)) throw new Error("a day is a date from 1900 to today");
  if (s && f && s > f) throw new Error("it cannot be finished before it was started");
  await own(db(), userId, id);
  await db().query("UPDATE learn_item SET started_on = $3, finished_on = $4, updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, s, f]);
}

/** Took part in a discussion or debate, or only heard it; how far through a course, of how many parts. */
export async function setItemFacts(userId: string, id: string, f: { tookPart?: boolean; part?: number | null; parts?: number | null }): Promise<void> {
  await tx(async (c) => {
    await own(c, userId, id);
    const cur = (await c.query("SELECT took_part, part, parts FROM learn_item WHERE id = $1", [id])).rows[0]!;
    const whole = (v: number | null | undefined, lo: number, name: string) => {
      if (v === null) return null;
      const n = Math.floor(Number(v));
      if (!(n >= lo && n <= 10000)) throw new Error(`${name} is a whole number from ${lo} to 10000`);
      return n;
    };
    const parts = f.parts !== undefined ? whole(f.parts, 1, "parts") : (cur.parts as number | null);
    let part = f.part !== undefined ? whole(f.part, 0, "part") : (cur.part as number | null);
    if (part !== null && parts !== null) part = Math.min(part, parts);
    const took = f.tookPart !== undefined ? Boolean(f.tookPart) : Boolean(cur.took_part);
    await c.query("UPDATE learn_item SET took_part = $2, part = $3, parts = $4, updated_at = now() WHERE id = $1", [id, took, part, parts]);
  });
}

/** Where the person is in a video or recording, in seconds, and the day they watched or heard it. */
export async function keepItemPlace(userId: string, id: string, seconds: number, day?: string, now = new Date()): Promise<void> {
  const s = Math.max(0, Math.floor(seconds));
  if (!Number.isFinite(s)) return;
  await tx(async (c) => {
    await own(c, userId, id);
    await c.query("UPDATE learn_item SET place_seconds = $3 WHERE id = $1 AND user_id = $2", [id, userId, s]);
    await keepDay(c, userId, today(day, now));
  });
}

/** Keeps a pasted transcript, sealed (empty clears it). */
export async function setTranscript(userId: string, id: string, text: string | null): Promise<void> {
  const t = clean(text, 400_000);
  await own(db(), userId, id);
  await db().query("UPDATE learn_item SET transcript_ct = $3, updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, t ? seal(userId, t) : null]);
}

/** A practice session: the day, how many minutes if given, and a note. */
export async function addSession(userId: string, id: string, s: { day?: string; minutes?: number | null; note?: string }, now = new Date()): Promise<void> {
  const day = s.day ? okDay(s.day, now) : today(undefined, now);
  if (!day) throw new Error("a day is a date from 1900 to today");
  const minutes = s.minutes ? Math.floor(s.minutes) : null;
  if (minutes !== null && !(minutes >= 1 && minutes <= 1440)) throw new Error("minutes are 1 to 1440");
  const note = clean(s.note, 2000);
  await tx(async (c) => {
    await own(c, userId, id);
    await c.query("INSERT INTO learn_item_session (id, item_id, user_id, day, minutes, note_ct) VALUES ($1, $2, $3, $4, $5, $6)", [randomUUID(), id, userId, day, minutes, note ? seal(userId, note) : null]);
    await keepDay(c, userId, day);
  });
}

export async function removeSession(userId: string, sessionId: string): Promise<void> {
  if (!UUID.test(sessionId)) return;
  await db().query("DELETE FROM learn_item_session WHERE id = $1 AND user_id = $2", [sessionId, userId]);
}

/** A note at a moment in a video or recording (seconds), with the words quoted from its transcript if any. */
export async function addItemNote(userId: string, id: string, n: { at?: number | null; quote?: string; note?: string; unit?: string | null }): Promise<string> {
  const quote = clean(n.quote, 4000);
  const note = clean(n.note, 4000);
  if (!quote && !note) throw new Error("a note needs words");
  const at = n.at === null || n.at === undefined || !Number.isFinite(n.at) ? null : Math.max(0, Math.floor(n.at));
  const nid = randomUUID();
  await own(db(), userId, id);
  await db().query("INSERT INTO learn_item_note (id, item_id, user_id, at_seconds, quote_ct, note_ct, unit) VALUES ($1, $2, $3, $4, $5, $6, $7)", [nid, id, userId, at, quote ? seal(userId, quote) : null, note ? seal(userId, note) : null, unitOf(n.unit)]);
  return nid;
}

export async function removeItemNote(userId: string, noteId: string): Promise<void> {
  if (!UUID.test(noteId)) return;
  await db().query("UPDATE learn_item_note SET removed_at = now() WHERE id = $1 AND user_id = $2", [noteId, userId]);
}

/** Subjects an item is about, like a book's. */
export async function addItemSubjects(userId: string, id: string, names: string[]): Promise<void> {
  await tx(async (c) => {
    await own(c, userId, id);
    for (const n of names.map((x) => x.replace(/\s+/g, " ").trim()).filter(Boolean).slice(0, 30)) {
      await c.query("INSERT INTO learn_item_subject (item_id, subject_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [id, await subjectId(c, userId, n)]);
    }
  });
}

export async function removeItemSubject(userId: string, id: string, subject: string): Promise<void> {
  await own(db(), userId, id);
  await db().query("DELETE FROM learn_item_subject WHERE item_id = $1 AND subject_id = $2", [id, subject]);
}

/** Ties a claim argued in a debate or discussion to it. The claim's holders are its sides. */
export async function linkItemClaim(userId: string, id: string, claimId: string): Promise<void> {
  if (!UUID.test(claimId)) throw new Error("no such claim");
  await own(db(), userId, id);
  const ok = await db().query("SELECT 1 FROM learn_claim WHERE id = $1 AND user_id = $2", [claimId, userId]);
  if (!ok.rowCount) throw new Error("no such claim");
  await db().query("INSERT INTO learn_item_claim (item_id, claim_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [id, claimId]);
}

const unitOf = (u: string | null | undefined) => (u && u.trim() ? u.trim().slice(0, 300) : null);

/** Where the person is in one unit of an item (a lecture, an episode), and the day they watched or heard it. */
export async function keepUnitPlace(userId: string, id: string, unit: string, seconds: number, day?: string, now = new Date()): Promise<void> {
  const u = unitOf(unit);
  const s = Math.max(0, Math.floor(seconds));
  if (!u || !Number.isFinite(s)) return;
  await tx(async (c) => {
    const row = await own(c, userId, id);
    await c.query(
      `INSERT INTO learn_item_unit (item_id, user_id, unit, place_seconds) VALUES ($1, $2, $3, $4)
       ON CONFLICT (item_id, unit) DO UPDATE SET place_seconds = $4, updated_at = now()`,
      [id, userId, u, s],
    );
    const d = today(day, now);
    if (row.status === "want") await c.query("UPDATE learn_item SET status = 'doing', started_on = COALESCE(started_on, $3::date), updated_at = now() WHERE id = $1 AND user_id = $2", [id, userId, d]);
    await keepDay(c, userId, d);
  });
}

/**
 * Marks a unit finished (or not). How many are finished is how far through the item is; finishing the last of the
 * units it has finishes the item, and finishing one starts an item not yet started.
 */
export async function setUnitDone(userId: string, id: string, unit: string, done: boolean, of: number | null, day?: string, now = new Date()): Promise<void> {
  const u = unitOf(unit);
  if (!u) return;
  await tx(async (c) => {
    const row = await own(c, userId, id);
    const d = today(day, now);
    await c.query(
      `INSERT INTO learn_item_unit (item_id, user_id, unit, done_on) VALUES ($1, $2, $3, $4)
       ON CONFLICT (item_id, unit) DO UPDATE SET done_on = $4, updated_at = now()`,
      [id, userId, u, done ? d : null],
    );
    const n = Number((await c.query("SELECT count(*) AS n FROM learn_item_unit WHERE item_id = $1 AND done_on IS NOT NULL", [id])).rows[0]!.n);
    const parts = of && of >= 1 && of <= 10000 ? Math.floor(of) : null;
    // Taking back a lecture of a finished course reopens it; a course with a lecture finished is under way.
    const status = parts && n >= parts ? "done" : !done && parts && row.status === "done" ? "doing" : n > 0 && row.status === "want" ? "doing" : row.status;
    await c.query(
      `UPDATE learn_item SET part = LEAST($3, COALESCE($4, parts, 10000)), parts = COALESCE($4, parts), status = $5,
       started_on = CASE WHEN $5 <> 'want' THEN COALESCE(started_on, $6::date) ELSE started_on END,
       finished_on = CASE WHEN $5 = 'done' THEN COALESCE(finished_on, $6::date) ELSE NULL END, updated_at = now()
       WHERE id = $1 AND user_id = $2`,
      [id, userId, n, parts, status, d],
    );
    if (done) await keepDay(c, userId, d);
  });
}

/** Each unit of an item the person has begun or finished. */
export async function listUnits(userId: string, id: string): Promise<ItemUnit[]> {
  if (!UUID.test(id)) return [];
  const r = await db().query("SELECT unit, place_seconds, done_on FROM learn_item_unit WHERE item_id = $1 AND user_id = $2", [id, userId]);
  return r.rows.map((x) => ({ unit: x.unit as string, place: x.place_seconds === null ? null : Number(x.place_seconds), doneOn: dayOf(x.done_on) }));
}

/** Removes an item, kept so it can be restored. */
export async function removeItem(userId: string, id: string): Promise<void> {
  if (!UUID.test(id)) return;
  await db().query("UPDATE learn_item SET removed_at = now() WHERE id = $1 AND user_id = $2", [id, userId]);
}

export async function restoreItem(userId: string, id: string): Promise<void> {
  if (!UUID.test(id)) return;
  await db().query("UPDATE learn_item SET removed_at = NULL WHERE id = $1 AND user_id = $2", [id, userId]);
}

type Row = Record<string, unknown>;
function itemOf(userId: string, x: Row, subjects: ItemSubject[], lesson: string | null): LearnItem {
  const body = JSON.parse(open(userId, x.body_ct as string)) as Body;
  return {
    id: x.id as string, medium: x.medium as Medium, way: isWay(x.way) ? x.way : null, status: isStatus(x.status) ? x.status : "done",
    title: body.title, by: body.by ?? null, url: body.url ?? null, image: body.image ?? null, work: body.work ?? null, lesson,
    startedOn: dayOf(x.started_on), finishedOn: dayOf(x.finished_on), tookPart: Boolean(x.took_part),
    part: x.part === null || x.part === undefined ? null : Number(x.part), parts: x.parts === null || x.parts === undefined ? null : Number(x.parts),
    place: x.place_seconds === null || x.place_seconds === undefined ? null : Number(x.place_seconds),
    subjects, transcript: Boolean(x.transcript_ct), at: new Date(x.created_at as string).toISOString(),
  };
}

/** Every item the person keeps, newest first; removed ones only when asked. */
export async function listItems(userId: string, opts: { removed?: boolean } = {}): Promise<LearnItem[]> {
  const r = await db().query(
    `SELECT i.*, (SELECT l.text_ct FROM learn_item_lesson l WHERE l.item_id = i.id ORDER BY l.created_at DESC LIMIT 1) AS last_lesson
     FROM learn_item i WHERE i.user_id = $1 AND (i.removed_at IS ${opts.removed ? "NOT NULL" : "NULL"}) ORDER BY i.created_at DESC`,
    [userId],
  );
  const subs = await db().query("SELECT s.item_id, s.subject_id, sub.name_ct FROM learn_item_subject s JOIN learn_item i ON i.id = s.item_id JOIN learn_subject sub ON sub.id = s.subject_id WHERE i.user_id = $1", [userId]).catch(() => ({ rows: [] as Row[] }));
  const byItem = new Map<string, ItemSubject[]>();
  for (const s of subs.rows) byItem.set(s.item_id as string, [...(byItem.get(s.item_id as string) ?? []), { id: s.subject_id as string, name: open(userId, s.name_ct as string) }]);
  return r.rows.filter((x) => isMedium(x.medium)).map((x) => itemOf(userId, x, (byItem.get(x.id as string) ?? []).sort((a, b) => a.name.localeCompare(b.name)), x.last_lesson ? open(userId, x.last_lesson as string) : null));
}

/** One item with its lessons, sessions, notes, claims and transcript. */
export async function getItem(userId: string, id: string): Promise<ItemFull | null> {
  if (!UUID.test(id)) return null;
  const items = await db().query("SELECT * FROM learn_item WHERE id = $1 AND user_id = $2", [id, userId]);
  const x = items.rows[0];
  if (!x || !isMedium(x.medium)) return null;
  const [lessons, sessions, notes, claims, subs] = await Promise.all([
    db().query("SELECT id, text_ct, created_at FROM learn_item_lesson WHERE item_id = $1 ORDER BY created_at", [id]),
    db().query("SELECT id, day, minutes, note_ct FROM learn_item_session WHERE item_id = $1 ORDER BY day DESC, created_at DESC", [id]),
    db().query("SELECT id, at_seconds, quote_ct, note_ct, created_at, unit FROM learn_item_note WHERE item_id = $1 AND removed_at IS NULL ORDER BY at_seconds NULLS LAST, created_at", [id]),
    db().query("SELECT claim_id FROM learn_item_claim WHERE item_id = $1", [id]),
    db().query("SELECT s.subject_id, sub.name_ct FROM learn_item_subject s JOIN learn_subject sub ON sub.id = s.subject_id WHERE s.item_id = $1", [id]),
  ]);
  const ls = lessons.rows.map((l) => ({ id: l.id as string, text: open(userId, l.text_ct as string), at: new Date(l.created_at as string).toISOString() }));
  return {
    ...itemOf(userId, x, subs.rows.map((s) => ({ id: s.subject_id as string, name: open(userId, s.name_ct as string) })).sort((a, b) => a.name.localeCompare(b.name)), ls.at(-1)?.text ?? null),
    lessons: ls,
    sessions: sessions.rows.map((s) => ({ id: s.id as string, day: dayOf(s.day)!, minutes: s.minutes === null ? null : Number(s.minutes), note: s.note_ct ? open(userId, s.note_ct as string) : null })),
    notes: notes.rows.map((n) => ({ id: n.id as string, at: n.at_seconds === null ? null : Number(n.at_seconds), quote: n.quote_ct ? open(userId, n.quote_ct as string) : null, note: n.note_ct ? open(userId, n.note_ct as string) : null, created: new Date(n.created_at as string).toISOString(), unit: (n.unit as string | null) ?? null })),
    claimIds: claims.rows.map((c) => c.claim_id as string),
    transcriptText: x.transcript_ct ? open(userId, x.transcript_ct as string) : null,
  };
}

/** What an older input (migration 0011: film, talk, conversation, link) is as an item. */
export function inputAsItem(b: Record<string, unknown>): { medium: Medium; title: string; by: string | null; url: string | null; status: ItemStatus; finishedOn: string | null; tookPart: boolean } | null {
  const s = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : null);
  const full = (v: unknown) => (typeof v === "string" && DAY.test(v) ? v : null);
  if (b.kind === "film") {
    const st = b.status === "want" ? "want" : b.status === "watching" ? "doing" : "done";
    return s(b.title) ? { medium: b.film === "series" ? "series" : "film", title: s(b.title)!, by: s(b.by), url: null, status: st, finishedOn: null, tookPart: false } : null;
  }
  if (b.kind === "talk") return s(b.title) ? { medium: "lecture", title: s(b.title)!, by: s(b.speaker), url: s(b.url), status: "done", finishedOn: full(b.on), tookPart: false } : null;
  if (b.kind === "conversation") {
    const who = Array.isArray(b.with) ? (b.with as unknown[]).map(s).filter(Boolean).join(", ") : null;
    return s(b.about) ? { medium: "discussion", title: s(b.about)!, by: who, url: null, status: "done", finishedOn: full(b.on), tookPart: true } : null;
  }
  if (b.kind === "link") {
    const medium: Medium = b.link === "video" || b.link === "channel" ? "lecture" : b.link === "course" ? "course" : b.link === "podcast" ? "podcast" : "article";
    return s(b.title) ? { medium, title: s(b.title)!, by: s(b.by), url: s(b.url), status: "done", finishedOn: null, tookPart: false } : null;
  }
  return null;
}

/**
 * Brings the older inputs (films, talks, conversations and links kept through MCP and ideas) into items, once:
 * each keeps its id, so an idea that names an input names its item, and its subjects come with it.
 */
export async function mergeInputs(userId: string): Promise<number> {
  const r = await db().query("SELECT i.id, i.body_ct, i.created_at FROM learn_input i LEFT JOIN learn_item t ON t.id = i.id WHERE i.user_id = $1 AND t.id IS NULL", [userId]);
  let n = 0;
  for (const x of r.rows) {
    const it = inputAsItem(JSON.parse(open(userId, x.body_ct as string)) as Record<string, unknown>);
    if (!it) continue;
    await tx(async (c) => {
      await c.query(
        "INSERT INTO learn_item (id, user_id, medium, status, body_ct, finished_on, took_part, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT (id) DO NOTHING",
        [x.id, userId, it.medium, it.status, seal(userId, JSON.stringify({ title: it.title, by: it.by, url: goodUrl(it.url), image: null, work: null })), it.finishedOn, it.tookPart, x.created_at],
      );
      await c.query("INSERT INTO learn_item_subject (item_id, subject_id) SELECT $1, subject_id FROM learn_input_subject WHERE input_id = $1 ON CONFLICT DO NOTHING", [x.id]);
    });
    n++;
  }
  return n;
}

/** Moves a book to another way, or back to the way its catalog gives (null). */
export async function setBookWay(userId: string, bookId: string, way: string | null): Promise<void> {
  await db().query("UPDATE learn_book SET way = $3 WHERE id = $1 AND user_id = $2", [bookId, userId, isWay(way) ? way : null]);
}
