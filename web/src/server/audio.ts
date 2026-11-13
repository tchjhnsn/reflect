import { librivoxBook, type Audiobook } from "@reflect/core/librivox";
import type { WorkAccess } from "@reflect/core/openlibrary";
import { librivoxId } from "./free";

const slow = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 }, signal: AbortSignal.timeout(6000) });

/**
 * A book's free LibriVox recording: the catalog's links, else the book found in LibriVox's catalog by title and
 * author when it is old enough to be free; null when there is none.
 */
export async function audiobookFor(book: { title: string; authors: readonly string[]; year?: number }, access: WorkAccess | null): Promise<Audiobook | null> {
  const oldEnough = access?.access === "public" || (book.year ?? 9999) < 1930;
  const found = !access?.librivox.length && oldEnough ? librivoxId(book) : null;
  const recordings = access?.librivox.length ? access.librivox : found ? [found] : [];
  return recordings.length ? librivoxBook(recordings, slow).catch(() => null) : null;
}
