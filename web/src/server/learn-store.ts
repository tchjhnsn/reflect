import { randomUUID } from "node:crypto";
import type { PoolClient } from "@neondatabase/serverless";
import { BOOK_STATUS, GRASP, cleanIsbn, isGrasp, namedSubjects, type BookStatus, type Grasp } from "@reflect/core/learning";
import type { CatalogEntry } from "@reflect/core/openlibrary";
import { db } from "@/db/pool";
import { open, seal, tag } from "./crypto";

/**
 * What a person knows, by their own account, in the app (Torian, 2026-11-11). Books are the center: the
 * person keeps a book and the subjects it covers, names what they remember of it without looking (a
 * recall), and rates their own grasp of each subject. Nothing is known until they rate it, the latest
 * grasp counts, and recalls and grasps are never changed.
 */

export type BookOut = { id: string; title: string; authors: string[]; isbn: string | null; status: BookStatus; startedOn: string | null; finishedOn: string | null; subjects: SubjectOut[]; catalog: CatalogEntry | null; recalls: { at: string; text: string }[] };
export type SubjectOut = { id: string; name: string; vouches: number; shown: number; suggested: number; grasp: { grasp: Grasp; at: string; bookId: string | null } | null };
export type LearnView = { known: SubjectOut[]; shown: SubjectOut[]; vouched: SubjectOut[]; suggested: SubjectOut[]; books: (BookOut & { unrated: string[] })[] };

const LIMIT = { title: 300, name: 120, text: 20_000, subjects: 60 };
const knows = (g: Grasp | undefined) => g === "teach" || g === "explain" || g === "gist";
const normName = (s: string) => s.replace(/\s+/g, " ").trim();

export async function tx<T>(f: (c: PoolClient) => Promise<T>): Promise<T> {
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    const out = await f(client);
    await client.query("COMMIT");
    return out;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

export async function subjectId(c: PoolClient, userId: string, name: string): Promise<string> {
  const n = normName(name);
  if (!n || n.length > LIMIT.name) throw new Error("a subject has a name of at most 120 characters");
  const t = tag(userId, n.toLowerCase());
  const found = await c.query("SELECT id FROM learn_subject WHERE user_id = $1 AND name_tag = $2", [userId, t]);
  if (found.rows[0]) return found.rows[0].id as string;
  const id = randomUUID();
  await c.query("INSERT INTO learn_subject (id, user_id, name_tag, name_ct) VALUES ($1, $2, $3, $4) ON CONFLICT (user_id, name_tag) DO NOTHING", [id, userId, t, seal(userId, n)]);
  const again = await c.query("SELECT id FROM learn_subject WHERE user_id = $1 AND name_tag = $2", [userId, t]);
  return again.rows[0].id as string;
}

export type BookInput = { title: string; authors?: string[]; isbn?: string; status?: BookStatus; subjects?: string[]; catalog?: CatalogEntry | null; on?: string };

/** Keep a new book. Its subjects are what the person chose; the catalog's are kept only as a suggestion. */
export async function addBook(userId: string, b: BookInput, now = new Date()): Promise<string> {
  const title = normName(b.title ?? "");
  if (!title || title.length > LIMIT.title) throw new Error("a book needs a title");
  const status = b.status ?? "want";
  if (!(BOOK_STATUS as readonly string[]).includes(status)) throw new Error(`status is one of ${BOOK_STATUS.join(", ")}`);
  const isbn = b.isbn ? cleanIsbn(b.isbn) : null;
  if (isbn && !/^(97[89])?\d{9}[\dX]$/.test(isbn)) throw new Error("an ISBN has 10 or 13 digits");
  if ((b.subjects?.length ?? 0) > LIMIT.subjects) throw new Error("at most 60 subjects on a book");
  const day = b.on ?? now.toISOString().slice(0, 10);
  const id = randomUUID();
  await tx(async (c) => {
    await c.query(
      "INSERT INTO learn_book (id, user_id, title_ct, authors_ct, isbn_ct, catalog_ct, status, started_on, finished_on) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
      [id, userId, seal(userId, title), seal(userId, JSON.stringify(b.authors ?? [])), isbn ? seal(userId, isbn) : null, b.catalog ? seal(userId, JSON.stringify(b.catalog)) : null, status, status === "reading" || status === "read" ? day : null, status === "read" ? day : null],
    );
    await keepStatus(c, userId, id, status, day);
    for (const s of b.subjects ?? []) if (normName(s)) await c.query("INSERT INTO learn_book_subject (book_id, subject_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [id, await subjectId(c, userId, s)]);
  });
  return id;
}

async function ownBook(c: PoolClient, userId: string, bookId: string): Promise<{ status: BookStatus; started_on: string | null }> {
  if (!/^[0-9a-f-]{36}$/.test(bookId)) throw new Error("no such book");
  const r = await c.query("SELECT status, started_on FROM learn_book WHERE id = $1 AND user_id = $2", [bookId, userId]);
  if (!r.rows[0]) throw new Error("no such book");
  return r.rows[0] as { status: BookStatus; started_on: string | null };
}

/** Change how far the person is with a book. Reading sets when they started; read sets when they finished. */
export async function setBookStatus(userId: string, bookId: string, status: BookStatus, now = new Date()): Promise<void> {
  if (!(BOOK_STATUS as readonly string[]).includes(status)) throw new Error(`status is one of ${BOOK_STATUS.join(", ")}`);
  const day = now.toISOString().slice(0, 10);
  await tx(async (c) => {
    const b = await ownBook(c, userId, bookId);
    await c.query("UPDATE learn_book SET status = $3, started_on = $4, finished_on = $5 WHERE id = $1 AND user_id = $2", [bookId, userId, status, b.started_on ?? (status === "reading" || status === "read" ? day : null), status === "read" ? day : null]);
    if (b.status !== status) await keepStatus(c, userId, bookId, status, day);
  });
}

/** Keep one status change in the book's history (migration 0013): never changed, the log reads it. */
async function keepStatus(c: PoolClient, userId: string, bookId: string, status: BookStatus, day: string): Promise<void> {
  await c.query("INSERT INTO learn_book_status (id, user_id, book_id, status, on_day) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, bookId, status, day]);
}

/** Add subjects the person chose to a book. */
export async function addSubjects(userId: string, bookId: string, names: string[]): Promise<void> {
  await tx(async (c) => {
    await ownBook(c, userId, bookId);
    for (const n of names) if (normName(n)) await c.query("INSERT INTO learn_book_subject (book_id, subject_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [bookId, await subjectId(c, userId, n)]);
  });
}

/** Keep what the person named of a book without looking, and say which of its subjects they named. */
export async function keepRecall(userId: string, bookId: string, recall: string): Promise<{ named: string[]; unnamed: string[] }> {
  const said = recall.trim();
  if (!said || said.length > LIMIT.text) throw new Error("name what you remember, in your own words");
  await tx(async (c) => {
    await ownBook(c, userId, bookId);
    await c.query("INSERT INTO learn_recall (id, user_id, book_id, text_ct) VALUES ($1, $2, $3, $4)", [randomUUID(), userId, bookId, seal(userId, said)]);
  });
  const book = await getBook(userId, bookId);
  const named = namedSubjects(said, book!.subjects.map((s) => ({ id: s.id, title: s.name })));
  return { named: book!.subjects.filter((s) => named.has(s.id)).map((s) => s.name), unnamed: book!.subjects.filter((s) => !named.has(s.id)).map((s) => s.name) };
}

/** The person's own grasp of a subject, maybe from a book. A new row every time; the latest counts. */
export async function rateSubject(userId: string, subject: string, grasp: string, opts: { bookId?: string; note?: string } = {}): Promise<void> {
  if (!isGrasp(grasp)) throw new Error(`grasp is one of ${GRASP.map((g) => g.id).join(", ")}`);
  if (opts.note && opts.note.length > LIMIT.text) throw new Error("the note is too long");
  await tx(async (c) => {
    if (opts.bookId) await ownBook(c, userId, opts.bookId);
    const sid = await subjectId(c, userId, subject);
    await c.query("INSERT INTO learn_grasp (id, user_id, subject_id, book_id, grasp, note_ct) VALUES ($1, $2, $3, $4, $5, $6)", [randomUUID(), userId, sid, opts.bookId ?? null, grasp, opts.note?.trim() ? seal(userId, opts.note.trim()) : null]);
  });
}

export async function subjects(userId: string): Promise<Map<string, SubjectOut>> {
  const r = await db().query(
    `SELECT s.id, s.name_ct, g.grasp, g.created_at, g.book_id, (SELECT count(*) FROM learn_vouch v WHERE v.subject_id = s.id)::int AS vouches,
       (SELECT count(*) FROM learn_subject_mark m WHERE m.subject_id = s.id AND m.kind = 'shown')::int AS shown,
       (SELECT count(*) FROM learn_subject_mark m WHERE m.subject_id = s.id AND m.kind = 'suggested')::int AS suggested FROM learn_subject s
     LEFT JOIN LATERAL (SELECT grasp, created_at, book_id FROM learn_grasp WHERE subject_id = s.id ORDER BY created_at DESC, id DESC LIMIT 1) g ON true
     WHERE s.user_id = $1`,
    [userId],
  );
  return new Map(r.rows.map((x) => [x.id as string, { id: x.id as string, name: open(userId, x.name_ct as string), vouches: Number(x.vouches ?? 0), shown: Number(x.shown ?? 0), suggested: Number(x.suggested ?? 0), grasp: x.grasp ? { grasp: x.grasp as Grasp, at: new Date(x.created_at as string).toISOString(), bookId: (x.book_id as string | null) ?? null } : null }]));
}

const dayOf = (d: unknown) => (d ? (d instanceof Date ? d.toISOString().slice(0, 10) : String(d).slice(0, 10)) : null);

async function books(userId: string, only?: string): Promise<BookOut[]> {
  const subs = await subjects(userId);
  const r = await db().query(`SELECT * FROM learn_book WHERE user_id = $1 ${only ? "AND id = $2" : ""} ORDER BY created_at DESC`, only ? [userId, only] : [userId]);
  const out: BookOut[] = [];
  for (const b of r.rows) {
    const links = await db().query("SELECT subject_id FROM learn_book_subject WHERE book_id = $1", [b.id]);
    const recalls = await db().query("SELECT text_ct, created_at FROM learn_recall WHERE book_id = $1 ORDER BY created_at", [b.id]);
    out.push({
      id: b.id,
      title: open(userId, b.title_ct),
      authors: JSON.parse(open(userId, b.authors_ct)) as string[],
      isbn: b.isbn_ct ? open(userId, b.isbn_ct) : null,
      status: b.status,
      startedOn: dayOf(b.started_on),
      finishedOn: dayOf(b.finished_on),
      subjects: links.rows.map((l) => subs.get(l.subject_id as string)!).filter(Boolean).sort((a, c) => a.name.localeCompare(c.name)),
      catalog: b.catalog_ct ? (JSON.parse(open(userId, b.catalog_ct)) as CatalogEntry) : null,
      recalls: recalls.rows.map((x) => ({ at: new Date(x.created_at as string).toISOString(), text: open(userId, x.text_ct as string) })),
    });
  }
  return out;
}

export async function getBook(userId: string, bookId: string): Promise<BookOut | null> {
  if (!/^[0-9a-f-]{36}$/.test(bookId)) return null;
  return (await books(userId, bookId))[0] ?? null;
}

/**
 * Known subjects, and every book with the subjects of a read book not rated since it was finished. Shown
 * and suggested are never known: shown waits for the person's rating, and suggested is only someone's
 * thought that a subject relates to them.
 */
export async function learnView(userId: string): Promise<LearnView> {
  const subs = [...(await subjects(userId)).values()];
  const all = await books(userId);
  return {
    known: subs.filter((s) => knows(s.grasp?.grasp)).sort((a, b) => a.name.localeCompare(b.name)),
    shown: subs.filter((s) => s.shown > 0 && !s.grasp).sort((a, b) => a.name.localeCompare(b.name)),
    vouched: subs.filter((s) => s.vouches > 0 && !s.grasp).sort((a, b) => a.name.localeCompare(b.name)),
    suggested: subs.filter((s) => s.suggested > 0 && !s.grasp && !s.shown && !s.vouches).sort((a, b) => a.name.localeCompare(b.name)),
    books: all.map((b) => ({
      ...b,
      unrated: b.status === "read" ? b.subjects.filter((s) => !s.grasp || (s.grasp.bookId !== b.id && b.finishedOn !== null && s.grasp.at.slice(0, 10) < b.finishedOn)).map((s) => s.name) : [],
    })),
  };
}
