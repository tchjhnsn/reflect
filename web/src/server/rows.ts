import { browseQuery, searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { THEMA } from "@reflect/core/thema";
import { LISTS, bestListMatch, listBookQuery } from "@reflect/core/lists";
import { listBooks, openLibrary } from "./browse";

/** One row of For you below the person's own rows: a title, where See all goes, and its books. */
export type Row = { key: string; title: string; href: string; books: CatalogEntry[] };

const ROW = 6;
const take = (books: CatalogEntry[]) => books.slice(0, ROW);

/** Old books most read in the open library, the pool the free rows are drawn from. */
const classics = () => browseQuery("ebook_access:public", openLibrary, { limit: 60, sort: "readinglog" });

/**
 * The rows For you shows after the person's own, in order, each made only when it is reached: books free to read
 * in Reflect, audiobooks free to hear in Reflect, a list, the classics, then the best rated books of every Thema
 * category, section by section. Rows are asked for a few at a time as the person scrolls.
 */
export const ROWS: readonly { key: string; title: string; href: string; make: () => Promise<CatalogEntry[]> }[] = [
  { key: "read", title: "Read free in Reflect", href: "/learn/browse?view=free", make: async () => take((await classics()).filter((b) => b.free?.read)) },
  { key: "listen", title: "Listen free in Reflect", href: "/learn/browse?view=free", make: async () => take((await classics()).filter((b) => b.free?.listen)) },
  ...LISTS.map((l) => ({
    key: `list-${l.id}`,
    title: `From ${l.name}`,
    href: `/learn/lists/${l.id}`,
    make: async () => (await Promise.all(l.books.slice(0, ROW).map((b) => searchBooks(listBookQuery(b), openLibrary, 6).then((r) => bestListMatch(b, r)).catch(() => null)))).filter((b): b is CatalogEntry => b !== null),
  })),
  { key: "before-1900", title: "Most read books written before 1900", href: "/learn/browse?q=first_publish_year%3A%5B*%20TO%201899%5D&sort=readinglog", make: async () => take(await browseQuery("first_publish_year:[* TO 1899]", openLibrary, { limit: ROW, sort: "readinglog" })) },
  ...THEMA.flatMap((s) => s.children.map((c) => ({ key: `cat-${c.code}`, title: `Best rated in ${c.short}`, href: `/learn/browse?cat=${c.code}`, make: async () => take(await listBooks({ kind: "cat", code: c.code }, 1, "rating")) }))),
];

/** Rows `from` to `from + n` of For you, made at once; a row the open library cannot fill is left out. */
export async function rowsAt(from: number, n: number): Promise<Row[]> {
  const slice = ROWS.slice(from, from + n);
  const made = await Promise.all(slice.map((r) => r.make().catch((): CatalogEntry[] => [])));
  return slice.flatMap((r, i) => (made[i]!.length ? [{ key: r.key, title: r.title, href: r.href, books: made[i]! }] : []));
}
