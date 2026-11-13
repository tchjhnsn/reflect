// Looking a book up in Open Library (openlibrary.org, the Internet Archive's open catalog; no key). What
// comes back is a catalog entry anyone can edit, so it is a suggestion: the person keeps a book's title,
// authors and subjects only by saying so. The entry is saved beside the book note as
// learn/books/<book>.openlibrary.json, which the recall check reads for the book's contents.
import { join } from "node:path";
import { BOOKS, cleanIsbn, slugOf } from "./learning.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export type Fetch = (url: string) => Promise<{ ok: boolean; status: number; json(): Promise<unknown> }>;
export type CatalogEntry = { work: string; title: string; authors: string[]; year?: number; isbn?: string; subjects: string[]; contents: string[]; lcc?: string[]; /** Open Library cover id. */ cover?: number; /** Open Library readers' star rating, 1 to 5, when shown in a list. */ rating?: { average: number; count: number } };

/** The cover image for an entry, from Open Library's covers service, or null when it has none. */
export const coverUrl = (e: Pick<CatalogEntry, "cover">, size: "S" | "M" | "L" = "M"): string | null => (e.cover ? `https://covers.openlibrary.org/b/id/${e.cover}-${size}.jpg` : null);

const BASE = "https://openlibrary.org";
const FIELDS = "key,title,author_name,first_publish_year,isbn,subject,lcc,cover_i,ratings_average,ratings_count";
type Doc = { key?: string; title?: string; author_name?: string[]; first_publish_year?: number; isbn?: string[]; subject?: string[]; lcc?: string[]; cover_i?: number; ratings_average?: number; ratings_count?: number };

async function get(fetcher: Fetch, path: string): Promise<unknown> {
  const r = await fetcher(`${BASE}${path}`);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Open Library answered ${r.status}`);
  return r.json();
}

/** A catalog subject list is long and mixed; keep the first distinct ones, in the catalog's order. */
function topSubjects(subjects: readonly string[] | undefined, n = 15): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of subjects ?? []) {
    const k = s.toLowerCase().trim();
    if (!k || seen.has(k) || /^(accessible book|protected daisy|in library|large type books|fiction, general|open library staff picks)$/i.test(k) || k.startsWith("nyt:")) continue;
    seen.add(k);
    out.push(s.trim());
    if (out.length >= n) break;
  }
  return out;
}

const fromDoc = (d: Doc, isbn?: string): CatalogEntry => ({
  work: (d.key ?? "").replace("/works/", ""),
  title: d.title ?? "",
  authors: d.author_name ?? [],
  ...(d.first_publish_year ? { year: d.first_publish_year } : {}),
  ...(isbn ? { isbn } : d.isbn?.find((i) => i.length === 13) ? { isbn: d.isbn.find((i) => i.length === 13) } : {}),
  subjects: topSubjects(d.subject),
  contents: [],
  ...(d.lcc?.length ? { lcc: d.lcc.slice(0, 20) } : {}),
  ...(Number.isInteger(d.cover_i) && d.cover_i! > 0 ? { cover: d.cover_i } : {}),
  ...(d.ratings_count && d.ratings_average ? { rating: { average: Math.round(d.ratings_average * 100) / 100, count: d.ratings_count } } : {}),
});

/** Books whose title or author matches, best first. */
export async function searchBooks(query: string, fetcher: Fetch, limit = 5, page = 1): Promise<CatalogEntry[]> {
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(query)}&fields=${FIELDS}&limit=${limit}${page > 1 ? `&page=${page}` : ""}`)) as { docs?: Doc[] } | null;
  return (body?.docs ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

/** What people are reading on Open Library this week. The trending list carries no call numbers; `workEntry` fills them in. */
export async function trendingBooks(fetcher: Fetch, limit = 24): Promise<CatalogEntry[]> {
  const body = (await get(fetcher, `/trending/weekly.json?limit=${Math.min(Math.max(limit, 1), 50)}`)) as { works?: Doc[] } | null;
  return (body?.works ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

/** One work by its id (OL...W), with its subjects and call numbers. */
export async function workEntry(work: string, fetcher: Fetch): Promise<CatalogEntry | null> {
  if (!/^OL\d+W$/.test(work)) return null;
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(`key:/works/${work}`)}&fields=${FIELDS}&limit=1`)) as { docs?: Doc[] } | null;
  const d = body?.docs?.[0];
  return d?.key && d.title ? fromDoc(d) : null;
}

/** Browse one Library of Congress main class (a letter, as in shelves.ts), best rated first. */
export async function browseClass(lccClass: string, fetcher: Fetch, opts: { page?: number; limit?: number } = {}): Promise<CatalogEntry[]> {
  if (!/^[A-Z]$/.test(lccClass)) throw new Error("a class is one capital letter of the Library of Congress Classification");
  const limit = Math.min(Math.max(opts.limit ?? 24, 1), 50);
  const page = Math.max(opts.page ?? 1, 1);
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(`lcc:${lccClass}*`)}&sort=rating&fields=${FIELDS}&limit=${limit}&page=${page}`)) as { docs?: Doc[] } | null;
  return (body?.docs ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

/**
 * Books matching an Open Library search query (as `themaQuery` writes them): best rated first, or, with
 * `sort: "readinglog"`, the ones most readers have logged, which keeps a narrow subject's list to known books.
 */
export async function browseQuery(query: string, fetcher: Fetch, opts: { page?: number; limit?: number; sort?: "rating" | "readinglog" } = {}): Promise<CatalogEntry[]> {
  const limit = Math.min(Math.max(opts.limit ?? 24, 1), 50);
  const page = Math.max(opts.page ?? 1, 1);
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(query)}&sort=${opts.sort ?? "rating"}&fields=${FIELDS}&limit=${limit}&page=${page}`)) as { docs?: Doc[] } | null;
  return (body?.docs ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

export type Edition = { title: string; year?: number; publisher?: string; isbn?: string; language?: string };
export type WorkDetails = { description: string | null; subjects: string[]; rating: { average: number; count: number } | null; editions: { count: number; first: Edition[] } };

/** What a book page shows beyond its catalog entry: the work's description, its readers' rating and its editions. */
export async function workDetails(work: string, fetcher: Fetch): Promise<WorkDetails | null> {
  if (!/^OL\d+W$/.test(work)) return null;
  const [w, r, e] = await Promise.all([
    get(fetcher, `/works/${work}.json`) as Promise<{ description?: string | { value?: string }; subjects?: string[] } | null>,
    get(fetcher, `/works/${work}/ratings.json`).catch(() => null) as Promise<{ summary?: { average?: number; count?: number } } | null>,
    get(fetcher, `/works/${work}/editions.json?limit=6`).catch(() => null) as Promise<{ size?: number; entries?: { title?: string; publish_date?: string; publishers?: string[]; isbn_13?: string[]; isbn_10?: string[]; languages?: { key?: string }[] }[] } | null>,
  ]);
  if (!w) return null;
  const raw = typeof w.description === "string" ? w.description : w.description?.value ?? "";
  // Descriptions are community-written Markdown; keep the text, drop link targets and the trailing source lines.
  const description = raw.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/\r/g, "").split(/\n-{3,}\n|\n\(\[source\]/)[0]!.trim().slice(0, 4000);
  const avg = r?.summary?.average;
  return {
    description: description || null,
    subjects: topSubjects(w.subjects, 20),
    rating: avg && r?.summary?.count ? { average: Math.round(avg * 100) / 100, count: r.summary.count } : null,
    editions: {
      count: e?.size ?? 0,
      first: (e?.entries ?? []).filter((x) => x.title).map((x) => {
        const year = Number(/\d{4}/.exec(x.publish_date ?? "")?.[0]);
        const isbn = x.isbn_13?.[0] ?? x.isbn_10?.[0];
        const language = x.languages?.[0]?.key?.replace("/languages/", "");
        return { title: x.title!, ...(year ? { year } : {}), ...(x.publishers?.[0] ? { publisher: x.publishers[0] } : {}), ...(isbn ? { isbn } : {}), ...(language ? { language } : {}) };
      }),
    },
  };
}

/** One edition by ISBN, with its work's subjects and, when the catalog has it, its table of contents. */
export async function lookupIsbn(isbn: string, fetcher: Fetch): Promise<CatalogEntry | null> {
  const n = cleanIsbn(isbn);
  const found = (await get(fetcher, `/search.json?q=isbn:${n}&fields=${FIELDS}&limit=1`)) as { docs?: Doc[] } | null;
  const doc = found?.docs?.[0];
  if (!doc?.key) return null;
  const entry = fromDoc(doc, n);
  const edition = (await get(fetcher, `/isbn/${n}.json`)) as { title?: string; table_of_contents?: ({ title?: string; level?: number } | string)[] } | null;
  entry.contents = (edition?.table_of_contents ?? []).map((c) => (typeof c === "string" ? c : `${"  ".repeat(c.level ?? 0)}${c.title ?? ""}`)).filter((c) => c.trim());
  return entry;
}

/** Save what the catalog said beside the book note. The book note itself changes only by the person. */
export async function saveCatalogEntry(root: string, book: string, entry: CatalogEntry, now: Date): Promise<string> {
  const rel = join(BOOKS, `${slugOf(book)}.openlibrary.json`);
  const old = await readRecord(root, rel);
  const at = now.toISOString();
  await writeRecord(root, rel, `${JSON.stringify({ source: `${BASE}/works/${entry.work}`, fetched_at: at, ...entry }, null, 2)}\n`, old ? "append" : "create", at);
  return rel;
}

export async function readCatalogEntry(root: string, book: string): Promise<CatalogEntry | null> {
  const t = await readRecord(root, join(BOOKS, `${slugOf(book)}.openlibrary.json`));
  return t ? (JSON.parse(t) as CatalogEntry) : null;
}

/**
 * A catalog entry that came back from a browser (a hidden form field, a chat card), cut to the fields and
 * sizes Reflect keeps. Returns null when it has no work id or title. Contents are never taken from a client.
 */
export function cleanEntry(raw: unknown): CatalogEntry | null {
  const e = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const strs = (v: unknown, n: number, max: number) => (Array.isArray(v) ? v.filter((x) => typeof x === "string" || typeof x === "number").map((x) => String(x).slice(0, max)).slice(0, n) : []);
  const work = typeof e.work === "string" ? e.work.trim().slice(0, 40) : "";
  const title = typeof e.title === "string" ? e.title.trim().slice(0, 300) : "";
  if (!work || !title) return null;
  const year = Number(e.year);
  const lcc = strs(e.lcc, 20, 60);
  const cover = Number(e.cover);
  return {
    work,
    title,
    authors: strs(e.authors, 10, 200),
    ...(Number.isInteger(year) && year > 0 ? { year } : {}),
    ...(typeof e.isbn === "string" && e.isbn.trim() ? { isbn: e.isbn.trim().slice(0, 20) } : {}),
    subjects: strs(e.subjects, 15, 200),
    contents: [],
    ...(lcc.length ? { lcc } : {}),
    ...(Number.isInteger(cover) && cover > 0 ? { cover } : {}),
  };
}
