import { workAccess } from "@reflect/core/openlibrary";
import { searchBook, type Hit } from "@reflect/core/reading";
import { learnView } from "./learn-store";
import { openLibrary } from "./browse";
import { readText, textSource } from "./texts";

/** The free books searched at once: the ones the person is reading first, then read, then want to read. */
export const PASSAGE_BOOKS = 16;
const ORDER: Record<string, number> = { reading: 0, read: 1, want: 2 };

export type PassageBook = { work: string; title: string; author: string; total: number; hits: Hit[]; chapters: string[] };

async function inBatches<T, R>(items: T[], n: number, f: (t: T) => Promise<R>): Promise<R[]> {
  const out: R[] = [];
  for (let i = 0; i < items.length; i += n) out.push(...(await Promise.all(items.slice(i, i + n).map(f))));
  return out;
}

/**
 * A phrase looked for in the whole text of each book on the person's shelves that can be read in Reflect (at most
 * `PASSAGE_BOOKS` of them), with where it occurs in each. Books without a free text, or whose text does not come,
 * are left out and counted.
 */
export async function searchShelves(userId: string, phrase: string): Promise<{ books: PassageBook[]; searched: number; unread: number }> {
  const view = await learnView(userId);
  const shelved = view.books
    .filter((b) => b.catalog?.work)
    .sort((a, b) => (ORDER[a.status] ?? 3) - (ORDER[b.status] ?? 3))
    .filter((b, i, all) => all.findIndex((x) => x.catalog!.work === b.catalog!.work) === i)
    .slice(0, PASSAGE_BOOKS * 2);
  let unread = 0;
  const found = await inBatches(shelved, 4, async (b) => {
    const work = b.catalog!.work;
    const access = await workAccess(work, openLibrary).catch(() => null);
    const source = await textSource({ title: b.catalog!.title, authors: b.catalog!.authors }, access).catch(() => null);
    if (!source) return null;
    const text = await readText(source, access?.gutenberg[0] ? { kind: "gutenberg", id: access.gutenberg[0] } : null).catch(() => null);
    if (!text) {
      unread++;
      return null;
    }
    const r = searchBook(text.chapters, phrase, 3);
    return { work, title: b.catalog!.title, author: b.catalog!.authors[0] ?? "", total: r.total, hits: r.hits, chapters: r.hits.map((h) => text.chapters[h.chapter - 1]?.title ?? "") };
  });
  const free = found.filter((x): x is PassageBook => x !== null).slice(0, PASSAGE_BOOKS);
  return { books: free.filter((b) => b.total > 0).sort((a, b) => b.total - a.total), searched: free.length, unread };
}
