// Standard Ebooks (standardebooks.org): public-domain books carefully proofread and typeset, free. This reads one
// book's whole text from its single-page edition for the reader inside Reflect, as chapters of plain paragraphs.
// Nothing is saved.
import type { Chapter, TextFetch } from "./gutenberg.ts";

/** A book's chapters and its endnotes by id ("note-1"). */
export type StandardBook = { id: string; chapters: Chapter[]; notes: Record<string, string> };

const text = (html: string) =>
  html
    .replace(/<br\s*\/?>/g, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;|&#160;/g, " ")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n: string) => String.fromCodePoint(Number(n)))
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();

const MARK = "\uE000";

/**
 * A paragraph's plain text with its note marks taken out: each mark (a link to "#note-N") is removed from the text
 * and kept as the character offset it sat at, so highlights count characters of the words alone.
 */
function withRefs(html: string): { plain: string; refs: { c: number; id: string }[] } {
  const ids: string[] = [];
  const marked = html.replace(/<a\b[^>]*href="#(note-\d+)"[^>]*>[\s\S]*?<\/a>/g, (_, id: string) => {
    ids.push(id);
    return MARK;
  });
  const t = text(marked);
  const refs: { c: number; id: string }[] = [];
  let plain = "";
  let k = 0;
  for (let i = 0; i < t.length; i++) {
    if (t[i] !== MARK) {
      plain += t[i];
      continue;
    }
    refs.push({ c: plain.length, id: ids[k++]! });
    // A mark between two spaces would leave two: keep one.
    if (plain.endsWith(" ") && t[i + 1] === " ") i++;
  }
  return { plain: plain.trim(), refs };
}

/** A Standard Ebooks edition's endnotes as plain text by id ("note-1"), the link back to the text left out. */
export function standardNotes(html: string): Record<string, string> {
  const out: Record<string, string> = {};
  const at = html.search(/<section\b[^>]*epub:type="[^"]*endnotes[^"]*"/);
  if (at < 0) return out;
  for (const m of html.slice(at).matchAll(/<li\b[^>]*id="(note-\d+)"[^>]*>([\s\S]*?)<\/li>/g)) {
    const t = text(m[2]!.replace(/<a\b[^>]*epub:type="backlink"[^>]*>[\s\S]*?<\/a>/g, ""));
    if (t) out[m[1]!] = t;
  }
  return out;
}

/**
 * A Standard Ebooks single-page edition as chapters: each section of the body matter that holds paragraphs, its
 * heading the text of its heading group ("I", or "Book I: The Childhood"). A bare Roman numeral in a chapter reads
 * "Chapter I". Exported for tests; `standardBook` fetches and calls it.
 */
export function standardChapters(html: string): Chapter[] {
  const out: Chapter[] = [];
  const opens = [...html.matchAll(/<section\b[^>]*epub:type="([^"]*)"[^>]*>/g)];
  for (const [i, m] of opens.entries()) {
    if (!/bodymatter/.test(m[1]!) && !/\bchapter\b/.test(m[1]!)) continue;
    const from = m.index! + m[0].length;
    const to = Math.min(...[opens[i + 1]?.index ?? html.length, html.indexOf("</section>", from)].filter((n) => n >= 0));
    const body = html.slice(from, to);
    const head = /<hgroup[^>]*>([\s\S]*?)<\/hgroup>/.exec(body)?.[1] ?? /<h[1-6][^>]*>([\s\S]*?)<\/h[1-6]>/.exec(body)?.[0] ?? "";
    const parts = [...head.matchAll(/<(?:h[1-6]|p)[^>]*>([\s\S]*?)<\/(?:h[1-6]|p)>/g)].map((x) => text(x[1]!)).filter(Boolean);
    const rest = body.replace(/<hgroup[^>]*>[\s\S]*?<\/hgroup>/, "").replace(/<h[1-6][^>]*>[\s\S]*?<\/h[1-6]>/, "");
    const paragraphs: string[] = [];
    const notes: { p: number; c: number; id: string }[] = [];
    for (const x of rest.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/g)) {
      const { plain, refs } = withRefs(x[1]!);
      if (!plain) continue;
      for (const r of refs) notes.push({ p: paragraphs.length, c: r.c, id: r.id });
      paragraphs.push(plain);
    }
    if (!paragraphs.length) continue;
    let title = parts.join(": ") || `Part ${out.length + 1}`;
    if (/^[IVXLCDM]+$/.test(title) && /chapter/.test(m[1]!)) title = `Chapter ${title}`;
    out.push(notes.length ? { title, paragraphs, notes } : { title, paragraphs });
  }
  return out;
}

/** One Standard Ebooks book (an id like "jane-austen/pride-and-prejudice") as chapters, or null. */
export async function standardBook(id: string, fetcher: TextFetch): Promise<StandardBook | null> {
  if (!/^[a-z0-9-]+(\/[a-z0-9-]+)+$/.test(id)) return null;
  const r = await fetcher(`https://standardebooks.org/ebooks/${id}/text/single-page`);
  if (!r.ok) return null;
  const html = await r.text();
  const chapters = standardChapters(html);
  return chapters.length ? { id, chapters, notes: standardNotes(html) } : null;
}
