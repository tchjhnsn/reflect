// A Goodreads export brought into Learn (@reflect/core/goodreads reads the file). Each book is looked up in
// Open Library so it lands on its Library of Congress shelf; a book already on the person's shelves (same
// ISBN, or same title) is left as it is. Lookups run a few at a time.
import { readGoodreads, type GoodreadsBook } from "@reflect/core/goodreads";
import { searchBooks, type CatalogEntry, type Fetch } from "@reflect/core/openlibrary";
import { addBook, learnView } from "./learn-store";

export const MAX_IMPORT_BOOKS = 500;
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

/** The catalog entry for a Goodreads book: by ISBN when it has one, else by title and author when the title matches. */
async function catalogFor(b: GoodreadsBook, fetcher: Fetch): Promise<CatalogEntry | null> {
  try {
    if (b.isbn) {
      const [hit] = await searchBooks(`isbn:${b.isbn}`, fetcher, 1);
      if (hit) return { ...hit, isbn: b.isbn };
    }
    const [hit] = await searchBooks(`${b.title} ${b.authors[0] ?? ""}`.trim(), fetcher, 1);
    const t = norm(b.title.replace(/\s*\(.*\)\s*$/, ""));
    return hit && (norm(hit.title).startsWith(t) || t.startsWith(norm(hit.title))) ? hit : null;
  } catch {
    return null;
  }
}

export type ImportReport = { added: number; already: number; uncataloged: number; overLimit: number };

export async function importGoodreads(userId: string, text: string, fetcher: Fetch, concurrency = 6): Promise<ImportReport> {
  const books = readGoodreads(text);
  const have = (await learnView(userId)).books;
  const isbns = new Set(have.map((b) => b.isbn).filter(Boolean));
  const titles = new Set(have.map((b) => norm(b.title)));
  const fresh = books.filter((b) => !(b.isbn && isbns.has(b.isbn)) && !titles.has(norm(b.title)));
  const take = fresh.slice(0, MAX_IMPORT_BOOKS);
  const report: ImportReport = { added: 0, already: books.length - fresh.length, uncataloged: 0, overLimit: fresh.length - take.length };
  let next = 0;
  const worker = async () => {
    while (next < take.length) {
      const b = take[next++]!;
      const catalog = await catalogFor(b, fetcher);
      if (!catalog) report.uncataloged++;
      await addBook(userId, { title: b.title, authors: b.authors, ...(b.isbn ? { isbn: b.isbn } : {}), status: b.status, catalog, ...(b.finishedOn ? { on: b.finishedOn } : {}) });
      report.added++;
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, take.length) }, worker));
  return report;
}
