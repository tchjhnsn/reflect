// Looking a book up in Open Library (openlibrary.org, the Internet Archive's open catalog; no key). What
// comes back is a catalog entry anyone can edit, so it is a suggestion: the person keeps a book's title,
// authors and subjects only by saying so. The entry is saved beside the book note as
// learn/books/<book>.openlibrary.json, which the recall check reads for the book's contents.
import { join } from "node:path";
import { BOOKS, cleanIsbn, slugOf } from "./learning.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export type Fetch = (url: string) => Promise<{ ok: boolean; status: number; json(): Promise<unknown> }>;
export type CatalogEntry = { work: string; title: string; authors: string[]; /** Open Library author ids (OL...A), in the order of `authors`. */ authorKeys?: string[]; year?: number; isbn?: string; subjects: string[]; contents: string[]; lcc?: string[]; /** Open Library cover id. */ cover?: number; /** Open Library readers' star rating, 1 to 5, when shown in a list. */ rating?: { average: number; count: number }; /** Whether the book can be read (its text) or heard (an audiobook) in Reflect, when the app has said so. */ free?: Free };
/** What of a book is free in Reflect: its text to read, an audiobook to hear. */
export type Free = { read?: true; listen?: true };

/** The cover image for an entry, from Open Library's covers service, or null when it has none. */
export const coverUrl = (e: Pick<CatalogEntry, "cover">, size: "S" | "M" | "L" = "M"): string | null => (e.cover ? `https://covers.openlibrary.org/b/id/${e.cover}-${size}.jpg` : null);

const BASE = "https://openlibrary.org";
/**
 * The language Reflect lists books in: English for now. Lists keep works with an edition in it, and each work
 * shows that edition's title and cover (a work's own title and cover are often another language's).
 */
export const LANGUAGE = { code: "eng", ui: "en", name: "English" } as const;
const FIELDS = "key,title,ebook_access,id_project_gutenberg,id_standard_ebooks,id_librivox,author_name,author_key,first_publish_year,isbn,subject,lcc,cover_i,ratings_average,ratings_count,language,editions,editions.key,editions.title,editions.cover_i,editions.language";
const LANG = `&lang=${LANGUAGE.ui}`;
/** A search kept to works with an edition in Reflect's language. */
const inLanguage = (q: string) => `(${q}) language:${LANGUAGE.code}`;
type EditionDoc = { key?: string; title?: string; cover_i?: number; language?: string[] };
type Doc = { key?: string; title?: string; author_name?: string[]; author_key?: string[]; first_publish_year?: number; isbn?: string[]; subject?: string[]; lcc?: string[]; cover_i?: number; ratings_average?: number; ratings_count?: number; language?: string[]; ebook_access?: string; id_project_gutenberg?: string[]; id_standard_ebooks?: string[]; id_librivox?: string[]; editions?: { docs?: EditionDoc[] } };

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

const plain = (t: string) => t.toLowerCase().replace(/[^a-z0-9]/g, "");
/** The edition in Reflect's language that a search picked for a work, when it has one. */
const localEdition = (d: Doc): EditionDoc | undefined => d.editions?.docs?.find((e) => e.language?.includes(LANGUAGE.code));
/** The work's title, unless the edition in Reflect's language is titled differently (a translation). */
function titleOf(d: Doc): string {
  const t = d.title ?? "";
  const e = localEdition(d)?.title?.trim().replace(/\.$/, "");
  return e && plain(e) !== plain(t) && !/^(the )?(harvard classics|works|collected works)\b/i.test(e) ? e : t;
}
function coverOf(d: Doc): number | undefined {
  const c = localEdition(d)?.cover_i ?? d.cover_i;
  return Number.isInteger(c) && c! > 0 ? c : undefined;
}

/** What the app says is free of a book in Reflect; none until it says (`knowFree`). */
let freeOf: ((b: { title: string; authors: string[]; year?: number; ebook?: string; gutenberg: boolean; standard: boolean; librivox: boolean }) => Free | null) | null = null;
/** Tell the catalog how to know whether a book can be read or heard in Reflect; every entry made after carries `free`. */
export function knowFree(f: typeof freeOf): void {
  freeOf = f;
}
function freeFor(d: Doc): Free | null {
  if (!freeOf) return null;
  const f = freeOf({ title: titleOf(d), authors: d.author_name ?? [], ...(d.first_publish_year ? { year: d.first_publish_year } : {}), ...(d.ebook_access ? { ebook: d.ebook_access } : {}), gutenberg: Boolean(d.id_project_gutenberg?.length), standard: Boolean(d.id_standard_ebooks?.length), librivox: Boolean(d.id_librivox?.length) });
  return f && (f.read || f.listen) ? f : null;
}

const fromDoc = (d: Doc, isbn?: string): CatalogEntry => {
  const free = freeFor(d);
  return {
  work: (d.key ?? "").replace("/works/", ""),
  title: titleOf(d),
  authors: d.author_name ?? [],
  ...(d.author_key?.length ? { authorKeys: d.author_key.filter((k) => AUTHOR_ID.test(k)).slice(0, 10) } : {}),
  ...(d.first_publish_year ? { year: d.first_publish_year } : {}),
  ...(isbn ? { isbn } : d.isbn?.find((i) => i.length === 13) ? { isbn: d.isbn.find((i) => i.length === 13) } : {}),
  subjects: topSubjects(d.subject),
  contents: [],
  ...(d.lcc?.length ? { lcc: d.lcc.slice(0, 20) } : {}),
  ...(coverOf(d) ? { cover: coverOf(d) } : {}),
  ...(d.ratings_count && d.ratings_average ? { rating: { average: Math.round(d.ratings_average * 100) / 100, count: d.ratings_count } } : {}),
  ...(free ? { free } : {}),
  };
};

/** An Open Library author id. */
export const AUTHOR_ID = /^OL\d+A$/;

/** The orders a list of books can be in: best rated, most read (most readers have logged it), newest, oldest. */
export const SORTS = { rating: "Best rated", readinglog: "Most read", new: "Newest", old: "Oldest" } as const;
export type Sort = keyof typeof SORTS;
export const isSort = (s: unknown): s is Sort => typeof s === "string" && Object.hasOwn(SORTS, s);

/**
 * Books whose title or author matches, best match first unless a sort is given, in Reflect's language. An
 * ISBN search finds any language.
 */
export async function searchBooks(query: string, fetcher: Fetch, limit = 5, page = 1, sort?: Sort): Promise<CatalogEntry[]> {
  const q = /^isbn:/i.test(query.trim()) ? query : inLanguage(query);
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(q)}${sort ? `&sort=${sort}` : ""}&fields=${FIELDS}${LANG}&limit=${limit}${page > 1 ? `&page=${page}` : ""}`)) as { docs?: Doc[] } | null;
  return (body?.docs ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

/**
 * What people are reading on Open Library this week, kept to works with an edition in Reflect's language and
 * shown by that edition. The trending list carries no call numbers; `workEntry` fills them in.
 */
export async function trendingBooks(fetcher: Fetch, limit = 24): Promise<CatalogEntry[]> {
  const n = Math.min(Math.max(limit, 1), 50);
  const body = (await get(fetcher, `/trending/weekly.json?limit=50`)) as { works?: Doc[] } | null;
  const works = (body?.works ?? []).filter((d) => d.key && d.title && d.language?.includes(LANGUAGE.code)).slice(0, n);
  if (!works.length) return [];
  // The trending list gives each work's own title and cover; one search gets the edition in Reflect's language.
  const keys = works.map((d) => `"${d.key}"`).join(" OR ");
  const found = (await get(fetcher, `/search.json?q=${encodeURIComponent(`key:(${keys})`)}&fields=${FIELDS}${LANG}&limit=${works.length}`).catch(() => null)) as { docs?: Doc[] } | null;
  const local = new Map((found?.docs ?? []).map((d) => [d.key, d]));
  return works.map((d) => fromDoc(local.get(d.key) ?? d));
}

/** One work by its id (OL...W), with its subjects and call numbers. */
export async function workEntry(work: string, fetcher: Fetch): Promise<CatalogEntry | null> {
  if (!/^OL\d+W$/.test(work)) return null;
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(`key:/works/${work}`)}&fields=${FIELDS}${LANG}&limit=1`)) as { docs?: Doc[] } | null;
  const d = body?.docs?.[0];
  return d?.key && d.title ? fromDoc(d) : null;
}

/** Browse one Library of Congress main class (a letter, as in shelves.ts), best rated first. */
export async function browseClass(lccClass: string, fetcher: Fetch, opts: { page?: number; limit?: number } = {}): Promise<CatalogEntry[]> {
  if (!/^[A-Z]$/.test(lccClass)) throw new Error("a class is one capital letter of the Library of Congress Classification");
  const limit = Math.min(Math.max(opts.limit ?? 24, 1), 50);
  const page = Math.max(opts.page ?? 1, 1);
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(inLanguage(`lcc:${lccClass}*`))}&sort=rating&fields=${FIELDS}${LANG}&limit=${limit}&page=${page}`)) as { docs?: Doc[] } | null;
  return (body?.docs ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

/**
 * Books matching an Open Library search query (as `themaQuery` writes them): best rated first, or, with
 * `sort: "readinglog"`, the ones most readers have logged, which keeps a narrow subject's list to known books.
 */
export async function browseQuery(query: string, fetcher: Fetch, opts: { page?: number; limit?: number; sort?: Sort } = {}): Promise<CatalogEntry[]> {
  const limit = Math.min(Math.max(opts.limit ?? 24, 1), 120);
  const page = Math.max(opts.page ?? 1, 1);
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(inLanguage(query))}&sort=${opts.sort ?? "rating"}&fields=${FIELDS}${LANG}&limit=${limit}&page=${page}`)) as { docs?: Doc[] } | null;
  return (body?.docs ?? []).filter((d) => d.key && d.title).map((d) => fromDoc(d));
}

export type Author = { id: string; name: string; birth?: string; death?: string; bio: string | null; photo: number | null; wikipedia: string | null };
export type AuthorHit = { id: string; name: string; topWork?: string; works: number; birth?: string; death?: string };

/** An author's photo from Open Library's covers service, or null when they have none. */
export const authorPhotoUrl = (photo: number | null, size: "S" | "M" | "L" = "M"): string | null => (photo ? `https://covers.openlibrary.org/a/id/${photo}-${size}.jpg` : null);

/** One author by id (OL...A): name, life dates, a short biography, a photo and their Wikipedia article. */
export async function authorDetails(id: string, fetcher: Fetch): Promise<Author | null> {
  if (!AUTHOR_ID.test(id)) return null;
  const a = (await get(fetcher, `/authors/${id}.json`)) as { name?: string; personal_name?: string; birth_date?: string; death_date?: string; bio?: string | { value?: string }; photos?: number[]; links?: { url?: string; title?: string }[]; remote_ids?: { wikidata?: string } } | null;
  if (!a || !(a.name ?? a.personal_name)) return null;
  const raw = typeof a.bio === "string" ? a.bio : a.bio?.value ?? "";
  const bio = raw.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/\r/g, "").split(/\n-{3,}\n|\n\(\[source\]|\n\(Source/)[0]!.trim().slice(0, 3000);
  const wiki = a.links?.find((l) => /^https?:\/\/en\.wikipedia\.org\//.test(l.url ?? ""))?.url?.replace(/^http:/, "https:") ?? null;
  return {
    id,
    name: (a.name ?? a.personal_name)!.trim(),
    ...(a.birth_date ? { birth: a.birth_date } : {}),
    ...(a.death_date ? { death: a.death_date } : {}),
    bio: bio || null,
    photo: a.photos?.find((p) => Number.isInteger(p) && p > 0) ?? null,
    wikipedia: wiki,
  };
}

/** An author's books in Reflect's language, best rated first unless another sort is given. */
export async function authorWorks(id: string, fetcher: Fetch, opts: { page?: number; limit?: number; sort?: Sort } = {}): Promise<CatalogEntry[]> {
  if (!AUTHOR_ID.test(id)) return [];
  return browseQuery(`author_key:${id}`, fetcher, opts);
}

/**
 * Authors whose name matches, most works first among the ones the catalog ranks well. An author with one
 * work is left out unless their name is the whole query, which drops most misspelled duplicates.
 */
export async function searchAuthors(query: string, fetcher: Fetch, limit = 4): Promise<AuthorHit[]> {
  const q = query.trim();
  if (!q || /^isbn:/i.test(q)) return [];
  const body = (await get(fetcher, `/search/authors.json?q=${encodeURIComponent(q)}&fields=key,name,top_work,work_count,birth_date,death_date&limit=10`)) as { docs?: { key?: string; name?: string; top_work?: string; work_count?: number; birth_date?: string; death_date?: string }[] } | null;
  return (body?.docs ?? [])
    .filter((d) => d.key && AUTHOR_ID.test(d.key) && d.name && ((d.work_count ?? 0) > 1 || plain(d.name) === plain(q)))
    .slice(0, limit)
    .map((d) => ({ id: d.key!, name: d.name!, ...(d.top_work ? { topWork: d.top_work } : {}), works: d.work_count ?? 0, ...(d.birth_date ? { birth: d.birth_date } : {}), ...(d.death_date ? { death: d.death_date } : {}) }));
}

export type Edition = { title: string; year?: number; publisher?: string; isbn?: string; language?: string; /** Open Library cover id. */ cover?: number; pages?: number };
export type WorkDetails = { description: string | null; subjects: string[]; rating: { average: number; count: number } | null; editions: { count: number; first: Edition[] }; /** The table of contents of the first edition in Reflect's language that has one. */ contents: string[] };

/** Catalog subjects too broad to make a list of books like another from. */
export const BROAD_SUBJECT = /^(fiction|nonfiction|history|biography|literature|juvenile fiction|juvenile literature|english language|large type books|textbooks|in library|accessible book|fiction, general|classic literature|novela|reading level-grade \d+)$/i;

/**
 * Where a work can be read or heard free: the Internet Archive's scan through Open Library (`public` to read,
 * `borrowable` to borrow with an Open Library account), Project Gutenberg, Standard Ebooks and LibriVox ids.
 */
export type WorkAccess = { access: "public" | "borrowable" | "printdisabled" | "no_ebook"; archive?: string; edition?: string; gutenberg: string[]; standardEbooks: string[]; librivox: string[] };

/** Where one work can be read or heard free, from the catalog's own links. */
export async function workAccess(work: string, fetcher: Fetch): Promise<WorkAccess | null> {
  if (!/^OL\d+W$/.test(work)) return null;
  const body = (await get(fetcher, `/search.json?q=${encodeURIComponent(`key:/works/${work}`)}&fields=key,ebook_access,lending_identifier_s,lending_edition_s,id_project_gutenberg,id_standard_ebooks,id_librivox&limit=1`)) as { docs?: { ebook_access?: string; lending_identifier_s?: string; lending_edition_s?: string; id_project_gutenberg?: string[]; id_standard_ebooks?: string[]; id_librivox?: string[] }[] } | null;
  const d = body?.docs?.[0];
  if (!d) return null;
  const ok = (v: string[] | undefined, re: RegExp) => (v ?? []).filter((x) => re.test(x)).slice(0, 5);
  const access = (["public", "borrowable", "printdisabled"] as const).find((a) => a === d.ebook_access) ?? "no_ebook";
  return {
    access,
    ...(d.lending_identifier_s && /^[\w.-]+$/.test(d.lending_identifier_s) ? { archive: d.lending_identifier_s } : {}),
    ...(d.lending_edition_s && /^OL\d+M$/.test(d.lending_edition_s) ? { edition: d.lending_edition_s } : {}),
    gutenberg: ok(d.id_project_gutenberg, /^\d+$/),
    standardEbooks: ok(d.id_standard_ebooks, /^[a-z0-9-]+\/[a-z0-9-]+$/),
    librivox: ok(d.id_librivox, /^\d+$/),
  };
}

/** Books like a work: the most read on its first subject that is not too broad, without the work itself. */
export async function booksLike(entry: Pick<CatalogEntry, "work" | "subjects">, fetcher: Fetch, limit = 6): Promise<CatalogEntry[]> {
  const subject = entry.subjects.find((s) => !BROAD_SUBJECT.test(s.trim()));
  if (!subject) return [];
  const books = await browseQuery(`subject:"${subject.replace(/"/g, "")}"`, fetcher, { limit: limit + 1, sort: "readinglog" });
  return books.filter((b) => b.work !== entry.work).slice(0, limit);
}

/** What a book page shows beyond its catalog entry: the work's description, its readers' rating and its editions. */
export async function workDetails(work: string, fetcher: Fetch): Promise<WorkDetails | null> {
  if (!/^OL\d+W$/.test(work)) return null;
  const [w, r, e] = await Promise.all([
    get(fetcher, `/works/${work}.json`) as Promise<{ description?: string | { value?: string }; subjects?: string[] } | null>,
    get(fetcher, `/works/${work}/ratings.json`).catch(() => null) as Promise<{ summary?: { average?: number; count?: number } } | null>,
    get(fetcher, `/works/${work}/editions.json?limit=30`).catch(() => null) as Promise<{ size?: number; entries?: { title?: string; publish_date?: string; publishers?: string[]; isbn_13?: string[]; isbn_10?: string[]; languages?: { key?: string }[]; covers?: number[]; number_of_pages?: number; table_of_contents?: ({ title?: string; level?: number } | string)[] }[] } | null>,
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
        const cover = x.covers?.find((c) => Number.isInteger(c) && c > 0);
        const pages = Number.isInteger(x.number_of_pages) && x.number_of_pages! > 0 ? x.number_of_pages : undefined;
        return { title: x.title!, ...(year ? { year } : {}), ...(x.publishers?.[0] ? { publisher: x.publishers[0] } : {}), ...(isbn ? { isbn } : {}), ...(language ? { language } : {}), ...(cover ? { cover } : {}), ...(pages ? { pages } : {}) };
      }).sort((a, b) => Number(b.language === LANGUAGE.code) - Number(a.language === LANGUAGE.code) || Number(Boolean(b.cover)) - Number(Boolean(a.cover))), // Reflect's language first, then editions with a cover
    },
    contents: ((e?.entries ?? []).find((x) => (x.languages?.[0]?.key ?? "/languages/eng") === `/languages/${LANGUAGE.code}` && (x.table_of_contents?.length ?? 0) > 1)?.table_of_contents ?? [])
      .map((c) => (typeof c === "string" ? c : `${"  ".repeat(Math.min(c.level ?? 0, 4))}${c.title ?? ""}`).replace(/\s+$/, ""))
      .filter((c) => c.trim())
      .slice(0, 200),
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
  const authorKeys = strs(e.authorKeys, 10, 20).filter((k) => AUTHOR_ID.test(k));
  const cover = Number(e.cover);
  return {
    work,
    title,
    authors: strs(e.authors, 10, 200),
    ...(authorKeys.length ? { authorKeys } : {}),
    ...(Number.isInteger(year) && year > 0 ? { year } : {}),
    ...(typeof e.isbn === "string" && e.isbn.trim() ? { isbn: e.isbn.trim().slice(0, 20) } : {}),
    subjects: strs(e.subjects, 15, 200),
    contents: [],
    ...(lcc.length ? { lcc } : {}),
    ...(Number.isInteger(cover) && cover > 0 ? { cover } : {}),
  };
}
