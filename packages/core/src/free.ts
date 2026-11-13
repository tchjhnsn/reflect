// Whether a book can be read or heard in Reflect, from two catalogs of free public-domain books: Project
// Gutenberg's English texts and LibriVox's English recordings. Each catalog is lines of "id|title|author" in lower
// case (the web app keeps them as files, refreshed by a script). Open Library links only a few of its works to
// either, so a book is matched by its title and author instead.

type Row = { id: string; title: Set<string>; author: Set<string>; translators: string };
/** A catalog indexed by each word of its titles. */
export type FreeIndex = Map<string, Row[]>;

const SMALL = new Set(["the", "a", "an", "of", "and", "on", "in", "to", "or", "for", "with", "by", "at", "from"]);
const wordsOf = (s: string) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);

/** A catalog's lines as an index. */
export function freeIndex(lines: string): FreeIndex {
  const index: FreeIndex = new Map();
  for (const line of lines.split("\n")) {
    const [id, title, author, translators] = line.split("|");
    if (!id || !title) continue;
    const row = { id, title: new Set(wordsOf(title)), author: new Set(wordsOf(author ?? "")), translators: translators?.trim() ?? "" };
    // Each row is also found by its id ("#2199"), which no title word can be.
    index.set(`#${id}`, [row]);
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

/** The catalog rows that are the same book as the row with this id (itself included), as `freeEditions` finds them. */
export function editionsOfId(index: FreeIndex, id: string): { id: string; translators: string }[] {
  const row = index.get(`#${id}`)?.[0];
  if (!row) return [];
  const want = [...row.title].filter((w) => !SMALL.has(w));
  if (!want.length) return [];
  const names = [...row.author].filter((w) => w.length >= 4);
  const list = index.get(want.reduce((a, b) => ((index.get(b)?.length ?? 0) < (index.get(a)?.length ?? 0) ? b : a))) ?? [];
  return list
    .filter((r) => want.every((w) => r.title.has(w)) && [...r.title].filter((w) => !SMALL.has(w)).length === want.length && names.some((n) => r.author.has(n)))
    .map((r) => ({ id: r.id, translators: r.translators }));
}

/**
 * Every catalog row that is the same book: the same title words (small words aside), no more and no fewer, by an
 * author sharing a name with the book's first author, with its translators when the catalog names them. A book
 * kept in several translations has a row for each.
 */
export function freeEditions(index: FreeIndex, book: { title: string; authors?: readonly string[] }): { id: string; translators: string }[] {
  const want = wordsOf(book.title.split(/[:;(]/)[0] ?? "").filter((w) => !SMALL.has(w));
  const names = wordsOf(book.authors?.[0] ?? "").filter((w) => w.length >= 4);
  if (!want.length || !names.length) return [];
  const list = index.get(want.reduce((a, b) => ((index.get(b)?.length ?? 0) < (index.get(a)?.length ?? 0) ? b : a))) ?? [];
  return list
    .filter((r) => want.every((w) => r.title.has(w)) && [...r.title].filter((w) => !SMALL.has(w)).length === want.length && names.some((n) => r.author.has(n)))
    .map((r) => ({ id: r.id, translators: r.translators }));
}
