import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { open, seal, tag } from "./crypto";

/**
 * Excerpts from books (migration 0020). A highlight is an excerpt marked in a text read in Reflect, with its
 * anchor: the text's source, the chapter, and the paragraph and character it starts and ends at. An excerpt
 * typed from a book without a text keeps a page or place instead. Removing deletes; nothing is edited.
 */

/** Where a highlight sits in a text read in Reflect: paragraph indexes and character offsets within them, end exclusive. */
export type Anchor = { source: string; chapter: number; chapterTitle: string; from: [number, number]; to: [number, number] };
export type ExcerptOut = { id: string; work: string; quote: string; anchor: Anchor | null; place: string | null; note: string | null; at: string };
export type ExcerptInput = { work: string; quote: string; anchor?: Anchor | null; place?: string | null; note?: string | null };

const WORK = /^OL\d+W$/;
const SOURCE = /^(standard|gutenberg):[a-z0-9/-]{1,120}$/;
const whole = (n: unknown, max: number) => typeof n === "number" && Number.isInteger(n) && n >= 0 && n <= max;

/** An anchor that came back from a browser, checked again, or null. */
export function cleanAnchor(raw: unknown): Anchor | null {
  const a = (raw && typeof raw === "object" ? raw : null) as Record<string, unknown> | null;
  if (!a || typeof a.source !== "string" || !SOURCE.test(a.source) || !whole(a.chapter, 10000) || (a.chapter as number) < 1) return null;
  const pair = (p: unknown): [number, number] | null => (Array.isArray(p) && p.length === 2 && whole(p[0], 100000) && whole(p[1], 1000000) ? [p[0] as number, p[1] as number] : null);
  const from = pair(a.from);
  const to = pair(a.to);
  if (!from || !to || to[0] < from[0] || (to[0] === from[0] && to[1] <= from[1])) return null;
  return { source: a.source, chapter: a.chapter as number, chapterTitle: typeof a.chapterTitle === "string" ? a.chapterTitle.slice(0, 200) : "", from, to };
}

/** Keep an excerpt: a quote of 1 to 5,000 characters, with its anchor in the text or a page or place the person gives, and an optional note. */
export async function addExcerpt(userId: string, e: ExcerptInput): Promise<string> {
  if (!WORK.test(e.work)) throw new Error("no such book");
  const quote = e.quote.replace(/\s+/g, " ").trim();
  if (!quote) throw new Error("an excerpt needs its words");
  if (quote.length > 5000) throw new Error("an excerpt is at most 5,000 characters");
  const anchor = e.anchor ? cleanAnchor(e.anchor) : null;
  if (e.anchor && !anchor) throw new Error("the highlight's place in the text is not valid");
  const place = e.place?.trim().slice(0, 100) || null;
  const note = e.note?.trim() || null;
  if (note && note.length > 5000) throw new Error("the note is too long");
  const id = randomUUID();
  await db().query(
    "INSERT INTO learn_excerpt (id, user_id, work_tag, work_ct, quote_ct, anchor_ct, place_ct, note_ct) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
    [id, userId, tag(userId, e.work), seal(userId, e.work), seal(userId, quote), anchor ? seal(userId, JSON.stringify(anchor)) : null, place ? seal(userId, place) : null, note ? seal(userId, note) : null],
  );
  return id;
}

/** The person's excerpts from one book, oldest first. */
export async function excerptsOf(userId: string, work: string): Promise<ExcerptOut[]> {
  if (!WORK.test(work)) return [];
  const r = await db().query("SELECT * FROM learn_excerpt WHERE user_id = $1 AND work_tag = $2 ORDER BY created_at", [userId, tag(userId, work)]);
  return r.rows.map((x) => ({
    id: x.id as string,
    work: open(userId, x.work_ct as string),
    quote: open(userId, x.quote_ct as string),
    anchor: x.anchor_ct ? (JSON.parse(open(userId, x.anchor_ct as string)) as Anchor) : null,
    place: x.place_ct ? open(userId, x.place_ct as string) : null,
    note: x.note_ct ? open(userId, x.note_ct as string) : null,
    at: new Date(x.created_at as string).toISOString(),
  }));
}

/** Remove one of the person's excerpts. */
export async function removeExcerpt(userId: string, id: string): Promise<void> {
  if (!/^[0-9a-f-]{36}$/.test(id)) throw new Error("no such excerpt");
  const r = await db().query("DELETE FROM learn_excerpt WHERE id = $1 AND user_id = $2", [id, userId]);
  if (!r.rowCount) throw new Error("no such excerpt");
}
