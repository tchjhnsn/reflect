/**
 * Edition rankings from Hardcover (hardcover.app), the one book catalog whose API counts readers and ratings per
 * edition. Only aggregate numbers are used, credited to Hardcover, as its API terms allow; no user's library,
 * review or list, and no user-uploaded cover image. A work is found by its editions' ISBNs, else by title and
 * author in Hardcover's search; its editions are then ranked by Hardcover readers.
 */

/** Runs one GraphQL query against Hardcover and gives back its `data`, or null; the caller adds the token. */
export type HardcoverQuery = (query: string, variables: Record<string, unknown>) => Promise<unknown>;

export type HardcoverEdition = { id: number; title: string; isbn?: string; year?: number; pages?: number; publisher?: string; format?: string; language?: string; translators: string[]; readers: number; rating?: number };
export type HardcoverBook = { id: number; slug: string; title: string; rating?: number; ratings: number; readers: number; editions: HardcoverEdition[] };

export const HARDCOVER_URL = "https://api.hardcover.app/v1/graphql";

/** The authorization header for a Hardcover token, as it is pasted with or without its "Bearer " prefix. */
export const hardcoverAuth = (token: string): string => `Bearer ${token.trim().replace(/^bearer\s+/i, "")}`;

const BY_ISBN = `query ($isbns: [String!]) { editions(where: {_or: [{isbn_13: {_in: $isbns}}, {isbn_10: {_in: $isbns}}]}, limit: 30) { book_id } }`;
const SEARCH = `query ($q: String!) { search(query: $q, query_type: "Book", per_page: 5, page: 1) { results } }`;
const BOOK = `query ($id: Int!) { books(where: {id: {_eq: $id}}, limit: 1) { id slug title rating ratings_count users_count editions(where: {users_count: {_gt: 0}}, order_by: {users_count: desc}, limit: 25) { id title subtitle isbn_13 isbn_10 release_year pages physical_format edition_format users_count rating publisher { name } language { language } cached_contributors } } }`;

const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : typeof v === "string" && v.trim() && Number.isFinite(Number(v)) ? Number(v) : undefined);
const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});
const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

/** The book most of the matched editions belong to. */
export function mostCommon(ids: readonly number[]): number | null {
  const n = new Map<number, number>();
  for (const id of ids) n.set(id, (n.get(id) ?? 0) + 1);
  let best: number | null = null;
  for (const [id, c] of n) if (best === null || c > n.get(best)!) best = id;
  return best;
}

/** The search hit that is this book: its title is the book's title, and an author matches when one is given. */
export function pickSearchHit(results: unknown, title: string, author?: string): number | null {
  const hits = (obj(results).hits as unknown[] | undefined) ?? [];
  const t = norm(title);
  const a = author ? norm(author).split(" ").at(-1) ?? "" : "";
  for (const h of hits) {
    const d = obj(obj(h).document);
    const id = num(d.id);
    const names = Array.isArray(d.author_names) ? d.author_names.filter((x): x is string => typeof x === "string").map(norm).join(" ") : "";
    if (id !== undefined && norm(str(d.title) ?? "") === t && (!a || names.includes(a))) return id;
  }
  return null;
}

/** Who translated an edition, from its contributor list. */
export function translatorsOf(contributors: unknown): string[] {
  const list = Array.isArray(contributors) ? contributors : [];
  return list
    .map(obj)
    .filter((c) => /translat/i.test(str(c.contribution) ?? ""))
    .map((c) => str(obj(c.author).name))
    .filter((x): x is string => Boolean(x));
}

/** One book with its editions from a `books` query row, most read first; editions in other languages last. */
export function bookOf(row: unknown, language = "English"): HardcoverBook | null {
  const b = obj(row);
  const id = num(b.id);
  if (id === undefined) return null;
  const editions = ((b.editions as unknown[] | undefined) ?? []).map(obj).flatMap((e): HardcoverEdition[] => {
    const eid = num(e.id);
    const title = str(e.title);
    if (eid === undefined || !title) return [];
    const subtitle = str(e.subtitle);
    const isbn = str(e.isbn_13) ?? str(e.isbn_10);
    const year = num(e.release_year);
    const pages = num(e.pages);
    const publisher = str(obj(e.publisher).name);
    const format = str(e.edition_format) ?? str(e.physical_format);
    const lang = str(obj(e.language).language);
    const rating = num(e.rating);
    return [{
      id: eid,
      title: subtitle ? `${title}: ${subtitle}` : title,
      ...(isbn ? { isbn } : {}),
      ...(year ? { year } : {}),
      ...(pages && pages > 0 ? { pages } : {}),
      ...(publisher ? { publisher } : {}),
      ...(format ? { format } : {}),
      ...(lang ? { language: lang } : {}),
      translators: translatorsOf(e.cached_contributors),
      readers: num(e.users_count) ?? 0,
      ...(rating && rating > 0 ? { rating: Math.round(rating * 100) / 100 } : {}),
    }];
  });
  editions.sort((x, y) => Number((y.language ?? language) === language) - Number((x.language ?? language) === language) || y.readers - x.readers);
  const rating = num(b.rating);
  return { id, slug: str(b.slug) ?? String(id), title: str(b.title) ?? "", ...(rating && rating > 0 ? { rating: Math.round(rating * 100) / 100 } : {}), ratings: num(b.ratings_count) ?? 0, readers: num(b.users_count) ?? 0, editions };
}

/**
 * A work's book on Hardcover with its editions ranked by readers: found by the ISBNs of its editions, else by its
 * title and first author. Null when Hardcover does not have it or does not answer.
 */
export async function hardcoverBook(work: { title: string; authors: readonly string[]; isbns: readonly string[] }, query: HardcoverQuery): Promise<HardcoverBook | null> {
  const isbns = [...new Set(work.isbns.map((i) => i.replace(/[^0-9X]/gi, "")).filter((i) => i.length === 10 || i.length === 13))].slice(0, 40);
  let id: number | null = null;
  if (isbns.length) {
    const d = obj(await query(BY_ISBN, { isbns }).catch(() => null));
    id = mostCommon(((d.editions as unknown[] | undefined) ?? []).map((e) => num(obj(e).book_id)).filter((x): x is number => x !== undefined));
  }
  if (id === null) {
    const d = obj(await query(SEARCH, { q: [work.title, work.authors[0]].filter(Boolean).join(" ") }).catch(() => null));
    id = pickSearchHit(obj(d.search).results, work.title, work.authors[0]);
  }
  if (id === null) return null;
  const d = obj(await query(BOOK, { id }).catch(() => null));
  return bookOf(((d.books as unknown[] | undefined) ?? [])[0]);
}
