import { randomUUID } from "node:crypto";
import { isCalTile, isRepeat, type CalEvent, type CalTile } from "@reflect/core/calendar";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";

// The calendar's events (migration 0030): sealed titles and notes, tagged with a tile, removable with undo.

const DAY = /^\d{4}-\d{2}-\d{2}$/;
const TIME = /^([01]\d|2[0-3]):[0-5]\d$/;
const UUID = /^[0-9a-f-]{36}$/;

export type EventInput = { title: unknown; tile?: unknown; day: unknown; starts?: unknown; ends?: unknown; repeat?: unknown; until?: unknown; note?: unknown; recordId?: unknown };

function clean(e: EventInput) {
  const title = typeof e.title === "string" ? e.title.replace(/\s+/g, " ").trim().slice(0, 200) : "";
  if (!title) throw new Error("Name the event.");
  const day = String(e.day ?? "");
  if (!DAY.test(day) || Number.isNaN(Date.parse(day))) throw new Error("Give the day.");
  const time = (v: unknown) => (typeof v === "string" && TIME.test(v) ? v : null);
  const starts = time(e.starts);
  const ends = starts ? time(e.ends) : null;
  if (starts && ends && ends <= starts) throw new Error("An event ends after it starts.");
  const until = typeof e.until === "string" && DAY.test(e.until) && e.until >= day ? e.until : null;
  const note = typeof e.note === "string" && e.note.trim() ? e.note.trim().slice(0, 4000) : null;
  return { title, tile: isCalTile(e.tile) ? e.tile : null, day, starts, ends, repeat: isRepeat(e.repeat) ? e.repeat : "none", until, note, recordId: typeof e.recordId === "string" && UUID.test(e.recordId) ? e.recordId : null };
}

export async function addEvent(userId: string, e: EventInput): Promise<string> {
  const c = clean(e);
  const id = randomUUID();
  await db().query(
    "INSERT INTO calendar_event (id, user_id, tile, title_ct, note_ct, day, starts, ends, repeat, until, record_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
    [id, userId, c.tile, seal(userId, c.title), c.note ? seal(userId, c.note) : null, c.day, c.starts, c.ends, c.repeat, c.until, c.recordId],
  );
  return id;
}

export async function saveEvent(userId: string, id: string, e: EventInput): Promise<void> {
  const c = clean(e);
  await db().query(
    "UPDATE calendar_event SET tile = $3, title_ct = $4, note_ct = $5, day = $6, starts = $7, ends = $8, repeat = $9, until = $10, updated_at = now() WHERE id = $1 AND user_id = $2",
    [id, userId, c.tile, seal(userId, c.title), c.note ? seal(userId, c.note) : null, c.day, c.starts, c.ends, c.repeat, c.until],
  );
}

export async function removeEvent(userId: string, id: string, removed: boolean): Promise<void> {
  await db().query(`UPDATE calendar_event SET removed_at = ${removed ? "now()" : "NULL"} WHERE id = $1 AND user_id = $2`, [id, userId]);
}

/** Events that can fall between two days: those starting by `to` that are repeating or start from `from` on. Filtered to a tile when given. */
export async function events(userId: string, from: string, to: string, tile?: CalTile | null): Promise<CalEvent[]> {
  const r = await db().query(
    `SELECT id, tile, title_ct, note_ct, day::text AS day, to_char(starts, 'HH24:MI') AS starts, to_char(ends, 'HH24:MI') AS ends, repeat, until::text AS until, record_id
     FROM calendar_event WHERE user_id = $1 AND removed_at IS NULL AND day <= $3 AND (day >= $2 OR (repeat <> 'none' AND (until IS NULL OR until >= $2)))${tile ? " AND tile = $4" : ""}`,
    tile ? [userId, from, to, tile] : [userId, from, to],
  );
  return r.rows.map((x) => ({
    id: x.id as string, tile: (x.tile as CalTile | null) ?? null, title: open(userId, x.title_ct as string), note: x.note_ct ? open(userId, x.note_ct as string) : null,
    day: x.day as string, starts: (x.starts as string | null) ?? null, ends: (x.ends as string | null) ?? null, repeat: x.repeat as CalEvent["repeat"], until: (x.until as string | null) ?? null, recordId: (x.record_id as string | null) ?? null,
  }));
}
