// Kindle highlights brought into Learn (@reflect/core/kindle reads My Clippings.txt). Each book is found in Open
// Library by title and author, and each highlight kept as an excerpt with its page and location and any note typed
// on it. A highlight already kept for that book (the same words) is skipped; a book the catalog cannot find is
// counted and left out. Lookups run a few at a time.
import { parseClippings, type Clipping } from "@reflect/core/kindle";
import { searchBooks, type Fetch } from "@reflect/core/openlibrary";
import { addExcerpt, excerptsOf } from "./excerpt-store";

export const MAX_CLIPPINGS = 5000;
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

export type KindleReport = { added: number; already: number; books: number; notFound: string[]; overLimit: number };

/** Open Library's work for a Kindle book, when the title found matches the Kindle's. */
async function workFor(title: string, author: string, fetcher: Fetch): Promise<string | null> {
  const t = norm(title.replace(/\s*[:(].*$/, ""));
  try {
    const [hit] = await searchBooks(`${title.replace(/\s*\(.*\)\s*$/, "")} ${author.split(/[;,]/)[0] ?? ""}`.trim(), fetcher, 1);
    return hit && (norm(hit.title).startsWith(t) || t.startsWith(norm(hit.title))) ? hit.work : null;
  } catch {
    return null;
  }
}

export async function importKindle(userId: string, text: string, fetcher: Fetch, concurrency = 4): Promise<KindleReport> {
  const all = parseClippings(text);
  const clips = all.slice(0, MAX_CLIPPINGS);
  const byBook = new Map<string, Clipping[]>();
  for (const c of clips) byBook.set(`${c.title}\u0000${c.author}`, [...(byBook.get(`${c.title}\u0000${c.author}`) ?? []), c]);
  const report: KindleReport = { added: 0, already: 0, books: byBook.size, notFound: [], overLimit: all.length - clips.length };
  const books = [...byBook.entries()];
  let next = 0;
  const worker = async () => {
    while (next < books.length) {
      const [, list] = books[next++]!;
      const { title, author } = list[0]!;
      const work = await workFor(title, author, fetcher);
      if (!work) {
        report.notFound.push(author ? `${title} (${author})` : title);
        continue;
      }
      const have = new Set((await excerptsOf(userId, work)).map((e) => norm(e.quote)));
      for (const c of list) {
        if (have.has(norm(c.text))) {
          report.already++;
          continue;
        }
        await addExcerpt(userId, { work, quote: c.text.slice(0, 5000), place: c.place || null, note: c.note ?? null });
        have.add(norm(c.text));
        report.added++;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, books.length) }, worker));
  return report;
}
