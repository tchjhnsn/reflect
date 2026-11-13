// Arithmetic for reading a book inside Reflect: how long a chapter takes, how far a place is, where a phrase
// occurs in the text, and a place carried from one text of a book to another. Pure: no I/O.
import type { Chapter } from "./gutenberg.ts";

/** Words a minute for reading prose silently (Brysbaert 2019, a review of 190 studies: 238 for non-fiction). */
export const WORDS_A_MINUTE = 238;

/** How many words a chapter's paragraphs hold. */
export function wordsOf(paragraphs: readonly string[]): number {
  let n = 0;
  for (const p of paragraphs) n += p.split(/\s+/).filter(Boolean).length;
  return n;
}

/** Minutes to read so many words, at least one when there are any. */
export function minutesFor(words: number): number {
  return words > 0 ? Math.max(1, Math.round(words / WORDS_A_MINUTE)) : 0;
}

/** "5 min" or "2 h 10 min". */
export function readingTime(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return h ? (m ? `${h} h ${m} min` : `${h} h`) : `${m} min`;
}

/**
 * Minutes left: the rest of this chapter (by how far through it the person is, 0 to 1) and the rest of the book
 * from here. `words` holds each chapter's words in order; `chapter` counts from 1.
 */
export function timeLeft(words: readonly number[], chapter: number, at: number): { chapter: number; book: number } {
  const here = words[chapter - 1] ?? 0;
  const restOfChapter = Math.round(here * (1 - Math.min(Math.max(at, 0), 1)));
  const after = words.slice(chapter).reduce((a, b) => a + b, 0);
  return { chapter: minutesFor(restOfChapter), book: minutesFor(restOfChapter + after) };
}

/** One place a phrase occurs: the chapter (from 1), the paragraph, the character offset, and the words around it. */
export type Hit = { chapter: number; p: number; c: number; before: string; match: string; after: string };

/**
 * Where a phrase occurs in a book, in reading order, ignoring case; at most `limit` places. A phrase shorter than
 * two characters finds nothing.
 */
export function searchBook(chapters: readonly Chapter[], phrase: string, limit = 100): { hits: Hit[]; total: number } {
  const q = phrase.replace(/\s+/g, " ").trim().toLowerCase();
  const hits: Hit[] = [];
  let total = 0;
  if (q.length < 2) return { hits, total };
  chapters.forEach((ch, ci) => {
    ch.paragraphs.forEach((p, pi) => {
      const low = p.toLowerCase();
      for (let k = low.indexOf(q); k >= 0; k = low.indexOf(q, k + q.length)) {
        total++;
        if (hits.length >= limit) continue;
        const a = Math.max(0, k - 70);
        const b = Math.min(p.length, k + q.length + 70);
        hits.push({ chapter: ci + 1, p: pi, c: k, before: (a > 0 ? "…" : "") + p.slice(a, k), match: p.slice(k, k + q.length), after: p.slice(k + q.length, b) + (b < p.length ? "…" : "") });
      }
    });
  });
  return { hits, total };
}

/** Each place a phrase occurs in one chapter's paragraphs, as spans, ignoring case. */
export function findSpans(paragraphs: readonly string[], phrase: string): { from: [number, number]; to: [number, number] }[] {
  const q = phrase.replace(/\s+/g, " ").trim().toLowerCase();
  if (q.length < 2) return [];
  const out: { from: [number, number]; to: [number, number] }[] = [];
  paragraphs.forEach((p, i) => {
    const low = p.toLowerCase();
    for (let k = low.indexOf(q); k >= 0; k = low.indexOf(q, k + q.length)) out.push({ from: [i, k], to: [i, k + q.length] });
  });
  return out;
}

/**
 * A place kept against one text of a book (with its chapter count) carried to a text with `n` chapters: the same
 * chapter when the counts agree, else the chapter at the same share of the book. Its position in the chapter is
 * kept only when the chapter is the same.
 */
export function carryPlace(place: { chapter: number; chapters: number; at: number }, n: number): { chapter: number; at: number } {
  if (n < 1) return { chapter: 1, at: 0 };
  if (place.chapters === n || place.chapters < 1) return { chapter: Math.min(Math.max(place.chapter, 1), n), at: place.at };
  const share = (place.chapter - 1 + place.at) / place.chapters;
  return { chapter: Math.min(n, Math.max(1, Math.floor(share * n) + 1)), at: 0 };
}
