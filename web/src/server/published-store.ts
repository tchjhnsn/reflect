import { randomBytes } from "node:crypto";
import { db } from "@/db/pool";

/**
 * Quotes the person publishes (migration 0021): a copy of one excerpt with its citation, made public only when
 * they say so, in plain text, at `/q/<id>`. Nothing else of theirs is public. Removing one takes its page down.
 */
export type PublishedQuote = { id: string; quote: string; author: string; title: string; place: string | null; work: string | null; at: string };

const ID = /^[A-Za-z0-9_-]{12}$/;
const out = (x: Record<string, unknown>): PublishedQuote => ({ id: x.id as string, quote: x.quote as string, author: x.author as string, title: x.title as string, place: (x.place as string | null) ?? null, work: (x.work as string | null) ?? null, at: new Date(x.created_at as string).toISOString() });

/** Publish a quote with its author, book and place. Returns its id. */
export async function publishQuote(userId: string, q: { quote: string; author: string; title: string; place?: string | null; work?: string | null }): Promise<string> {
  const quote = q.quote.replace(/\s+/g, " ").trim();
  const title = q.title.trim().slice(0, 300);
  if (!quote || !title) throw new Error("a quote needs its words and its book");
  if (quote.length > 2000) throw new Error("a published quote is at most 2,000 characters");
  const work = q.work && /^OL\d+W$/.test(q.work) ? q.work : null;
  const id = randomBytes(9).toString("base64url");
  await db().query("INSERT INTO published_quote (id, user_id, quote, author, title, place, work) VALUES ($1, $2, $3, $4, $5, $6, $7)", [id, userId, quote, q.author.trim().slice(0, 200), title, q.place?.trim().slice(0, 200) || null, work]);
  return id;
}

/** One published quote, for anyone. */
export async function publishedQuote(id: string): Promise<PublishedQuote | null> {
  if (!ID.test(id)) return null;
  const r = await db().query("SELECT * FROM published_quote WHERE id = $1", [id]);
  return r.rows[0] ? out(r.rows[0]) : null;
}

/** The quotes the person has published, newest first. */
export async function publishedBy(userId: string): Promise<PublishedQuote[]> {
  const r = await db().query("SELECT * FROM published_quote WHERE user_id = $1 ORDER BY created_at DESC", [userId]);
  return r.rows.map(out);
}

/** Take a published quote down. */
export async function unpublishQuote(userId: string, id: string): Promise<void> {
  if (!ID.test(id)) throw new Error("no such quote");
  const r = await db().query("DELETE FROM published_quote WHERE id = $1 AND user_id = $2", [id, userId]);
  if (!r.rowCount) throw new Error("no such quote");
}
