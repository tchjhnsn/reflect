// Project Gutenberg (gutenberg.org): public-domain books as plain text, free, with no key. This reads one
// book's text for the reader inside Reflect: the Gutenberg header and licence footer cut off, the hard-wrapped
// lines joined into paragraphs, and the text split into its chapters. Nothing is saved.
/** A fetch that can read a body as text. */
export type TextFetch = (url: string) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export type Chapter = { title: string; paragraphs: string[] };
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
