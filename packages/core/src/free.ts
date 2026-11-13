// Whether a book can be read or heard in Reflect, from two catalogs of free public-domain books: Project
// Gutenberg's English texts and LibriVox's English recordings. Each catalog is lines of "id|title|author" in lower
// case (the web app keeps them as files, refreshed by a script). Open Library links only a few of its works to
// either, so a book is matched by its title and author instead.

type Row = { id: string; title: Set<string>; author: Set<string> };
/** A catalog indexed by each word of its titles. */
export type FreeIndex = Map<string, Row[]>;

const SMALL = new Set(["the", "a", "an", "of", "and", "on", "in", "to", "or", "for", "with", "by", "at", "from"]);
const wordsOf = (s: string) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);

/** A catalog's lines as an index. */
export function freeIndex(lines: string): FreeIndex {
  const index: FreeIndex = new Map();
  for (const line of lines.split("\n")) {
    const [id, title, author] = line.split("|");
    if (!id || !title) continue;
    const row = { id, title: new Set(wordsOf(title)), author: new Set(wordsOf(author ?? "")) };
    for (const w of row.title) {
      if (SMALL.has(w)) continue;
      const list = index.get(w);
      if (list) list.push(row);
      else index.set(w, [row]);
    }
  }
  return index;
}

/**
 * The catalog id of a book: the first row whose title holds every word of the book's title (small words aside),
 * by an author sharing a name of four letters or more with the book's first author. A book with no author is not
 * matched: a title alone names too many books.
 */
export function findFree(index: FreeIndex, book: { title: string; authors?: readonly string[] }): string | null {
  const want = wordsOf(book.title.split(/[:;(]/)[0] ?? "").filter((w) => !SMALL.has(w));
  const names = wordsOf(book.authors?.[0] ?? "").filter((w) => w.length >= 4);
  if (!want.length || !names.length) return null;
  const lists = want.map((w) => index.get(w) ?? []);
  if (lists.some((l) => !l.length)) return null;
  const rarest = lists.reduce((a, b) => (b.length < a.length ? b : a));
  return rarest.find((r) => want.every((w) => r.title.has(w)) && names.some((n) => r.author.has(n)))?.id ?? null;
}
