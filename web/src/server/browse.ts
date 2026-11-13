import { AUTHOR_ID, authorWorks, browseQuery, searchBooks, type CatalogEntry, type Sort } from "@reflect/core/openlibrary";
import { themaCategory, themaQuery, themaSection } from "@reflect/core/thema";

/** What a list of books on Browse or an author page is drawn from: a search, a Thema section or category, or an author. */
export type BookSource = { kind: "q"; q: string } | { kind: "cat"; code: string } | { kind: "author"; id: string };

export const PER = 24;
export const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 3600 } });

/** A source that came back from a browser, checked again. */
export function cleanSource(raw: unknown): BookSource | null {
  const s = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  if (s.kind === "q" && typeof s.q === "string" && s.q.trim()) return { kind: "q", q: s.q.trim().slice(0, 200) };
  if (s.kind === "cat" && typeof s.code === "string" && (themaCategory(s.code) || themaSection(s.code))) return { kind: "cat", code: s.code.toUpperCase() };
  if (s.kind === "author" && typeof s.id === "string" && AUTHOR_ID.test(s.id)) return { kind: "author", id: s.id };
  return null;
}

/** One page of a source's books, `PER` to a page. A category is best rated first unless another sort is given. */
export async function listBooks(source: BookSource, page: number, sort?: Sort): Promise<CatalogEntry[]> {
  if (source.kind === "q") return searchBooks(source.q, openLibrary, PER, page, sort);
  if (source.kind === "author") return authorWorks(source.id, openLibrary, { limit: PER, page, ...(sort ? { sort } : {}) });
  const query = themaQuery(source.code);
  return query ? browseQuery(query, openLibrary, { limit: PER, page, sort: sort ?? "rating" }) : [];
}
