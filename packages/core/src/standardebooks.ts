// Standard Ebooks (standardebooks.org): public-domain books carefully proofread and typeset, free. This reads one
// book's whole text from its single-page edition for the reader inside Reflect, as chapters of plain paragraphs.
// Nothing is saved.
import type { Chapter, TextFetch } from "./gutenberg.ts";

export type StandardBook = { id: string; chapters: Chapter[] };

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
    const paragraphs = [...rest.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/g)].map((x) => text(x[1]!)).filter(Boolean);
    if (!paragraphs.length) continue;
    let title = parts.join(": ") || `Part ${out.length + 1}`;
    if (/^[IVXLCDM]+$/.test(title) && /chapter/.test(m[1]!)) title = `Chapter ${title}`;
    out.push({ title, paragraphs });
  }
  return out;
}

/** One Standard Ebooks book (an id like "jane-austen/pride-and-prejudice") as chapters, or null. */
export async function standardBook(id: string, fetcher: TextFetch): Promise<StandardBook | null> {
  if (!/^[a-z0-9-]+(\/[a-z0-9-]+)+$/.test(id)) return null;
  const r = await fetcher(`https://standardebooks.org/ebooks/${id}/text/single-page`);
  if (!r.ok) return null;
  const chapters = standardChapters(await r.text());
  return chapters.length ? { id, chapters } : null;
}
