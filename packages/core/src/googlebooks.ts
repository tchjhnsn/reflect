// Google Books (books.google.com), for reading: a preview of a book, or the whole of it when it is free, shown
// in Google's own reader on the book page, which searches inside the book. It needs an API key
// (GOOGLE_BOOKS_API_KEY, free from Google Cloud); without one the book page has no reader. Google's ratings
// are not used: on most books they come from a handful of readers. What comes back is shown beside the book
// and never saved.
import type { Fetch } from "./openlibrary.ts";

/** Google's reader for one edition: `full` when the whole book is free to read, a preview otherwise. */
export type GoogleRead = { id: string; full: boolean };
export type GoogleVolume = { read: GoogleRead | null; description: string | null; categories: string[]; link: string | null };
type Info = { title?: string; authors?: string[]; language?: string; description?: string; categories?: string[]; infoLink?: string };
type Item = { id?: string; volumeInfo?: Info; accessInfo?: { viewability?: string; embeddable?: boolean } };

const ARTICLES = new Set(["the", "a", "an"]);
const words = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);
const bare = (s: string) => {
  const w = words(s);
  return ARTICLES.has(w[0] ?? "") ? w.slice(1) : w;
};

/**
 * The same book: the same title, leading articles aside, or the title followed by "of" and the author ("The
 * Confessions of S. Augustin"), and, when the author is known, written by them. A workbook, a summary or a
 * book about the book is not the same book.
 */
export function sameBook(book: { title: string; authors?: readonly string[] }, v: Info): boolean {
  const want = bare(book.title);
  const got = bare(v.title ?? "");
  if (!want.length || got.slice(0, want.length).join(" ") !== want.join(" ")) return false;
  if (got.length > want.length && got[want.length] !== "of") return false;
  const names = words(book.authors?.[0] ?? "").filter((w) => w.length > 3).map((w) => w.slice(0, 6));
  return !names.length || (v.authors ?? []).some((a) => words(a).some((w) => names.some((n) => w.startsWith(n))));
}

const readable = (i: Item): GoogleRead | null =>
  i.id && /^[\w-]+$/.test(i.id) && i.accessInfo?.embeddable && (i.accessInfo.viewability === "ALL_PAGES" || i.accessInfo.viewability === "PARTIAL")
    ? { id: i.id, full: i.accessInfo.viewability === "ALL_PAGES" }
    : null;

/**
 * One book in Google Books: its edition found by ISBN, or else by title and first author with the title
 * checked. The reader comes from the edition with the most to read among the English editions of the same
 * book: the whole book free first, then a preview.
 */
export async function googleVolume(book: { title: string; authors?: readonly string[]; isbn?: string }, fetcher: Fetch, key: string | undefined): Promise<GoogleVolume | null> {
  if (!key) return null;
  const ask = async (q: string, n: number): Promise<Item[]> => {
    const r = await fetcher(`https://www.googleapis.com/books/v1/volumes?q=${encodeURIComponent(q)}&maxResults=${n}&printType=books&country=US&key=${encodeURIComponent(key)}`);
    if (!r.ok) throw new Error(`Google Books answered ${r.status}`);
    return ((await r.json()) as { items?: Item[] }).items ?? [];
  };
  const isbnHit = book.isbn ? (await ask(`isbn:${book.isbn.replace(/[^0-9X]/gi, "")}`, 1))[0] : undefined;
  const hitRead = isbnHit ? readable(isbnHit) : null;
  const same = hitRead?.full
    ? []
    : (await ask(`intitle:${book.title}${book.authors?.[0] ? ` inauthor:${book.authors[0]}` : ""}`, 20)).filter((i) => i.volumeInfo && sameBook(book, i.volumeInfo) && (!i.volumeInfo.language || i.volumeInfo.language === "en"));
  const v = isbnHit?.volumeInfo ?? same[0]?.volumeInfo;
  if (!v) return null;
  const reads = [hitRead, ...same.map(readable)].filter((r): r is GoogleRead => r !== null);
  return {
    read: reads.find((r) => r.full) ?? reads[0] ?? null,
    description: v.description ? v.description.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 4000) : null,
    categories: (v.categories ?? []).slice(0, 5),
    link: v.infoLink?.startsWith("https://") ? v.infoLink : v.infoLink?.startsWith("http://") ? v.infoLink.replace("http://", "https://") : null,
  };
}

/** Google's reader for an edition, framed in the page at the start of the book. It has its own search inside the book. */
export function readerUrl(id: string): string {
  return `https://books.google.com/books?id=${encodeURIComponent(id)}&pg=PP1&output=embed`;
}
