import { randomUUID } from "node:crypto";
import { ITEM_STATUS, isMedium, isWay, type ItemStatus, type Medium, type Way } from "@reflect/core/ways";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";

/**
 * Learn items (migration 0023): a lecture, debate, discussion, podcast, film, series, theatre, music, art or
 * game the person learned from, or keeps for rest. The title, maker and link are sealed together; what it taught
 * them is sealed apart. The way is the person's choice, or null to follow the medium.
 */
export type LearnItem = { id: string; medium: Medium; way: Way | null; status: ItemStatus; title: string; by: string | null; url: string | null; lesson: string | null; at: string };
export type ItemInput = { medium: string; way?: string | null; status?: string; title: string; by?: string; url?: string; lesson?: string };

const isStatus = (s: unknown): s is ItemStatus => typeof s === "string" && (ITEM_STATUS as readonly string[]).includes(s);
const clean = (s: string | undefined | null, max: number) => (s ?? "").trim().slice(0, max) || null;
const goodUrl = (u: string | null) => (u && /^https?:\/\/\S+$/i.test(u) ? u : null);

export async function addItem(userId: string, i: ItemInput): Promise<string> {
  const title = clean(i.title, 300);
  if (!title) throw new Error("Give it a title.");
  if (!isMedium(i.medium)) throw new Error("Unknown kind.");
  const id = randomUUID();
  const body = { title, by: clean(i.by, 200), url: goodUrl(clean(i.url, 1000)) };
  const lesson = clean(i.lesson, 4000);
  await db().query("INSERT INTO learn_item (id, user_id, medium, way, status, body_ct, lesson_ct) VALUES ($1, $2, $3, $4, $5, $6, $7)", [
    id, userId, i.medium, isWay(i.way) ? i.way : null, isStatus(i.status) ? i.status : "done", seal(userId, JSON.stringify(body)), lesson ? seal(userId, lesson) : null,
  ]);
  return id;
}

/** Changes the way (null follows the medium), the status, or what it taught them (empty clears it). */
export async function changeItem(userId: string, id: string, c: { way?: string | null; status?: string; lesson?: string | null }): Promise<void> {
  const sets: string[] = [];
  const args: unknown[] = [id, userId];
  if (c.way !== undefined) { args.push(isWay(c.way) ? c.way : null); sets.push(`way = $${args.length}`); }
  if (c.status !== undefined && isStatus(c.status)) { args.push(c.status); sets.push(`status = $${args.length}`); }
  if (c.lesson !== undefined) { const l = clean(c.lesson, 4000); args.push(l ? seal(userId, l) : null); sets.push(`lesson_ct = $${args.length}`); }
  if (!sets.length) return;
  await db().query(`UPDATE learn_item SET ${sets.join(", ")}, updated_at = now() WHERE id = $1 AND user_id = $2`, args);
}

export async function removeItem(userId: string, id: string): Promise<void> {
  await db().query("DELETE FROM learn_item WHERE id = $1 AND user_id = $2", [id, userId]);
}

export async function listItems(userId: string): Promise<LearnItem[]> {
  const r = await db().query("SELECT * FROM learn_item WHERE user_id = $1 ORDER BY created_at DESC", [userId]);
  return r.rows.filter((x) => isMedium(x.medium)).map((x) => {
    const body = JSON.parse(open(userId, x.body_ct as string)) as { title: string; by?: string | null; url?: string | null };
    return {
      id: x.id as string,
      medium: x.medium as Medium,
      way: isWay(x.way) ? x.way : null,
      status: isStatus(x.status) ? x.status : "done",
      title: body.title,
      by: body.by ?? null,
      url: body.url ?? null,
      lesson: x.lesson_ct ? open(userId, x.lesson_ct as string) : null,
      at: new Date(x.created_at as string).toISOString(),
    };
  });
}

/** Moves a book to another way, or back to the way its Thema category gives (null). */
export async function setBookWay(userId: string, bookId: string, way: string | null): Promise<void> {
  await db().query("UPDATE learn_book SET way = $3 WHERE id = $1 AND user_id = $2", [bookId, userId, isWay(way) ? way : null]);
}
