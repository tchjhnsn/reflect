// Google Books (books.google.com), for ratings and descriptions where Open Library's are thin. It needs an
// API key (GOOGLE_BOOKS_API_KEY, free from Google Cloud); without one Reflect shows Open Library's alone.
// What comes back is shown beside the book and never saved.
import type { Fetch } from "./openlibrary.ts";

export type GoogleVolume = { rating: { average: number; count: number } | null; description: string | null; categories: string[]; link: string | null };
type Item = { volumeInfo?: { title?: string; averageRating?: number; ratingsCount?: number; description?: string; categories?: string[]; infoLink?: string } };

const words = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);
const sameTitle = (a: string, b: string) => {
  const x = words(a).slice(0, 4).join(" ");
  return x.length > 0 && words(b).join(" ").startsWith(x);
};

/** One book's volume, found by ISBN, or else by title and first author with the title checked. */
export async function googleVolume(book: { title: string; authors?: readonly string[]; isbn?: string }, fetcher: Fetch, key: string | undefined): Promise<GoogleVolume | null> {
  if (!key) return null;
  const q = book.isbn ? `isbn:${book.isbn.replace(/[^0-9X]/gi, "")}` : `intitle:${book.title}${book.authors?.[0] ? ` inauthor:${book.authors[0]}` : ""}`;
  const r = await fetcher(`https://www.googleapis.com/books/v1/volumes?q=${encodeURIComponent(q)}&maxResults=3&printType=books&key=${encodeURIComponent(key)}`);
  if (!r.ok) throw new Error(`Google Books answered ${r.status}`);
  const items = ((await r.json()) as { items?: Item[] }).items ?? [];
  const v = (book.isbn ? items[0] : items.find((i) => sameTitle(book.title, i.volumeInfo?.title ?? "")))?.volumeInfo;
  if (!v) return null;
  return {
    rating: v.averageRating && v.ratingsCount ? { average: v.averageRating, count: v.ratingsCount } : null,
    description: v.description ? v.description.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 4000) : null,
    categories: (v.categories ?? []).slice(0, 5),
    link: v.infoLink?.startsWith("https://") ? v.infoLink : v.infoLink?.startsWith("http://") ? v.infoLink.replace("http://", "https://") : null,
  };
}
