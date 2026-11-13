import { workEntry } from "@reflect/core/openlibrary";
import type { ExcerptOut } from "./excerpt-store";

const cached = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 } });

/** One book's excerpts with its title and author, for the highlights page and export. */
export type HighlightBook = { work: string; title: string; author: string; cover?: number; excerpts: (ExcerptOut & { where: string | null })[] };

/** Where an excerpt is: its chapter for a highlight, else the page or place typed with it. */
export const whereOf = (e: ExcerptOut): string | null => (e.anchor ? e.anchor.chapterTitle || `Chapter ${e.anchor.chapter}` : e.place);

/**
 * The person's excerpts grouped by book, each book's title and author from the open library (cached a day), books
 * ordered by their newest excerpt, excerpts in reading order: by chapter and place, typed ones after, oldest first.
 */
export async function highlightBooks(_userId: string, excerpts: ExcerptOut[]): Promise<HighlightBook[]> {
  const works = [...new Set(excerpts.map((e) => e.work))];
  const entries = new Map(await Promise.all(works.map(async (w) => [w, await workEntry(w, cached).catch(() => null)] as const)));
  return works.map((w) => {
    const entry = entries.get(w);
    const list = excerpts
      .filter((e) => e.work === w)
      .sort((a, b) => (a.anchor && b.anchor ? a.anchor.chapter - b.anchor.chapter || a.anchor.from[0] - b.anchor.from[0] || a.anchor.from[1] - b.anchor.from[1] : a.anchor ? -1 : b.anchor ? 1 : a.at.localeCompare(b.at)))
      .map((e) => ({ ...e, where: whereOf(e) }));
    return { work: w, title: entry?.title ?? w, author: entry?.authors[0] ?? "", ...(entry?.cover ? { cover: entry.cover } : {}), excerpts: list };
  });
}
