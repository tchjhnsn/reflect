import { AUTHOR_ID, authorWorks, browseQuery, searchBooks, type CatalogEntry, type Sort } from "@reflect/core/openlibrary";
import { themaCategory, themaQuery, themaSection } from "@reflect/core/thema";

/** What a list of books on Browse or an author page is drawn from: a search, a Thema section or category, or an author. */
export type BookSource = { kind: "q"; q: string } | { kind: "cat"; code: string } | { kind: "author"; id: string } | { kind: "new" };

/** Books first published this year or last: the query behind New this year. */
export const newQuery = () => `first_publish_year:[${new Date().getUTCFullYear() - 1} TO *]`;

export const PER = 24;

/** Narrowing a list: books free to read, books from the last two years, books from before 1924 (in the public domain in the United States). */
export const FILTERS = { free: "Free to read", recent: "Published in the last two years", old: "Published before 1924" } as const;
export type Filter = keyof typeof FILTERS;
export const isFilter = (f: unknown): f is Filter => typeof f === "string" && Object.hasOwn(FILTERS, f);
function clause(f: Filter | undefined): string {
  if (f === "free") return " ebook_access:public";
  if (f === "recent") return ` ${newQuery()}`;
  if (f === "old") return " first_publish_year:[* TO 1923]";
  return "";
}
export const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 3600 } });

/** A source that came back from a browser, checked again. */
export function cleanSource(raw: unknown): BookSource | null {
  const s = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  if (s.kind === "q" && typeof s.q === "string" && s.q.trim()) return { kind: "q", q: s.q.trim().slice(0, 200) };
  if (s.kind === "cat" && typeof s.code === "string" && (themaCategory(s.code) || themaSection(s.code))) return { kind: "cat", code: s.code.toUpperCase() };
  if (s.kind === "author" && typeof s.id === "string" && AUTHOR_ID.test(s.id)) return { kind: "author", id: s.id };
  if (s.kind === "new") return { kind: "new" };
  return null;
}

/** One page of a source's books, `PER` to a page. A category is best rated first unless another sort is given. */
export async function listBooks(source: BookSource, page: number, sort?: Sort, filter?: Filter): Promise<CatalogEntry[]> {
  const f = clause(filter);
  if (source.kind === "q") return /^isbn:/i.test(source.q) ? searchBooks(source.q, openLibrary, PER, page, sort) : searchBooks(`${source.q}${f}`, openLibrary, PER, page, sort);
  if (source.kind === "author") return f ? browseQuery(`author_key:${source.id}${f}`, openLibrary, { limit: PER, page, ...(sort ? { sort } : {}) }) : authorWorks(source.id, openLibrary, { limit: PER, page, ...(sort ? { sort } : {}) });
  if (source.kind === "new") return browseQuery(`${newQuery()}${f}`, openLibrary, { limit: PER, page, sort: sort ?? "readinglog" });
  const query = themaQuery(source.code);
  return query ? browseQuery(f ? `(${query})${f}` : query, openLibrary, { limit: PER, page, sort: sort ?? "rating" }) : [];
}
