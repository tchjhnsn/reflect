// Project Gutenberg (gutenberg.org): public-domain books as plain text, free, with no key. This reads one
// book's text for the reader inside Reflect: the Gutenberg header and licence footer cut off, the hard-wrapped
// lines joined into paragraphs, and the text split into its chapters. Nothing is saved.
/** A fetch that can read a body as text. */
export type TextFetch = (url: string) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/** A chapter of plain paragraphs. `notes` are the note marks in it: paragraph, character offset, and the note's id in the book's notes. */
export type Chapter = { title: string; paragraphs: string[]; notes?: { p: number; c: number; id: string }[] };
export type GutenbergBook = { id: string; chapters: Chapter[] };

const HEADING = /^(chapter|book|part|letter|volume|act|canto|stave|section)\s+([ivxlcdm]+|\d+|the\s+\w+|[a-z]+(-[a-z]+)?)\b[.:]?(\s.{0,80})?$/i;
const ORDINAL = /^the\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last)\s+(book|part|chapter|letter)\.?$/i;
const MAX = 3_000_000;

/** Whether a paragraph is a chapter's heading, and the heading's text. Illustration captions that end in one count. */
function headingOf(p: string): string | null {
  const t = p.trim().replace(/\]$/, "");
  if (t.length <= 100 && (HEADING.test(t) || ORDINAL.test(t))) return t.replace(/[.\]]+$/, "");
  const cap = /^\[Illustration[^\]]*?((?:chapter|book|part)\s+[ivxlcdm\d]+)\.?\]$/i.exec(p.trim());
  return cap ? cap[1]! : null;
}

const ROMAN = /^(?=[ivxlcdm])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$/;
/** An all-capitals heading in title case, its Roman numerals kept: "CHAPTER II. OF MONEY" reads "Chapter II. Of Money". */
function titleCase(h: string): string {
  if (h !== h.toUpperCase()) return h;
  return h.toLowerCase().replace(/\b([a-z])([a-z']*)/g, (w, a: string, rest: string) => (ROMAN.test(w) ? w.toUpperCase() : a.toUpperCase() + rest));
}

/** A Gutenberg plain text as chapters. Exported for tests; `gutenbergBook` fetches and calls it. */
export function chaptersOf(text: string): Chapter[] {
  const start = text.search(/\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n/i);
  const end = text.search(/\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG/i);
  const body = text.slice(start >= 0 ? text.indexOf("\n", start) + 1 : 0, end >= 0 ? end : undefined).replace(/\r/g, "");
  const paragraphs = body.split(/\n\s*\n/).map((p) => p.split("\n").map((l) => l.trim()).join(" ").replace(/\s+/g, " ").replace(/_([^_]+)_/g, "$1").replace(/--/g, "\u2014").trim()).filter(Boolean);
  const sections: Chapter[] = [{ title: "Opening", paragraphs: [] }];
  for (const p of paragraphs) {
    const h = headingOf(p);
    if (h) sections.push({ title: titleCase(h), paragraphs: [] });
    else if (!/^\[Illustration/i.test(p)) sections.at(-1)!.paragraphs.push(p);
  }
  // A heading with almost nothing under it (a book's title before its first chapter, a list of contents) joins the next.
  const out: Chapter[] = [];
  let carry: string[] = [];
  for (const [i, s] of sections.entries()) {
    const last = i === sections.length - 1;
    if (s.paragraphs.length < 2 && !last && s.title !== "Opening") {
      carry.push(s.title);
      continue;
    }
    if (s.title === "Opening" && s.paragraphs.join(" ").length < 400) continue;
    out.push({ title: [...carry, s.title].join(" · "), paragraphs: s.paragraphs });
    carry = [];
  }
  // A text with no chapter headings is cut into parts of about forty paragraphs.
  if (out.length <= 1 && paragraphs.length > 60) {
    const parts: Chapter[] = [];
    for (let i = 0; i < paragraphs.length; i += 40) parts.push({ title: `Part ${parts.length + 1}`, paragraphs: paragraphs.slice(i, i + 40) });
    return parts;
  }
  return out;
}

/** One Gutenberg book's text as chapters, or null when Gutenberg has no plain text for it. */
export async function gutenbergBook(id: string, fetcher: TextFetch): Promise<GutenbergBook | null> {
  if (!/^\d{1,6}$/.test(id)) return null;
  const r = await fetcher(`https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt`);
  if (!r.ok) return null;
  const text = (await r.text()).slice(0, MAX);
  const chapters = chaptersOf(text);
  return chapters.length ? { id, chapters } : null;
}

/** One book in Gutenberg's catalog search: its number, title and author. */
export type GutenbergHit = { id: string; title: string; author: string };

const decode = (s: string) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&amp;/g, "&");

/** The books in an answer from Gutenberg's catalog search (its OPDS feed), in its order. Exported for tests. */
export function opdsHits(xml: string): GutenbergHit[] {
  const out: GutenbergHit[] = [];
  for (const e of xml.split("<entry>").slice(1)) {
    const id = /<id>https?:\/\/www\.gutenberg\.org\/ebooks\/(\d{1,6})\.opds<\/id>/.exec(e)?.[1];
    const title = /<title>([^<]*)<\/title>/.exec(e)?.[1];
    if (id && title) out.push({ id, title: decode(title).trim(), author: decode(/<content type="text">([^<]*)<\/content>/.exec(e)?.[1] ?? "").trim() });
  }
  return out;
}

const SMALL = new Set(["the", "a", "an", "of", "and", "on", "in", "to"]);
const wordsOf = (s: string) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);

/**
 * Whether a catalog hit is the book: every word of the book's title (small words aside) is in the hit's title,
 * which may be longer ("An Inquiry into the Nature and Causes of the Wealth of Nations"), and, when the author is
 * known, a word of their name of four letters or more is in the hit's author. Exported for tests.
 */
export function isTheBook(book: { title: string; authors?: readonly string[] }, hit: GutenbergHit): boolean {
  const want = wordsOf(book.title).filter((w) => !SMALL.has(w));
  const got = new Set(wordsOf(hit.title));
  if (!want.length || !want.every((w) => got.has(w))) return false;
  const names = (book.authors ?? []).flatMap((a) => wordsOf(a)).filter((w) => w.length >= 4);
  if (!names.length) return true;
  const by = wordsOf(hit.author);
  return names.some((n) => by.includes(n));
}

/** The Gutenberg number of a book, found in Gutenberg's own catalog by title and author, or null. Open Library links only some of them. */
export async function findGutenberg(book: { title: string; authors?: readonly string[] }, fetcher: TextFetch): Promise<string | null> {
  const last = (book.authors?.[0] ?? "").split(/\s+/).filter((w) => w.length >= 3).at(-1) ?? "";
  const q = `${book.title.replace(/[:;(].*$/, "")} ${last}`.trim();
  if (!q) return null;
  const r = await fetcher(`https://www.gutenberg.org/ebooks/search.opds/?query=${encodeURIComponent(q)}`);
  if (!r.ok) return null;
  return opdsHits(await r.text()).find((h) => isTheBook(book, h))?.id ?? null;
}
