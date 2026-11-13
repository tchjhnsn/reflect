// Highlights in a text read in Reflect: finding a kept highlight in the text again, and cutting a paragraph into
// the runs of plain and highlighted text the reader draws. A highlight keeps its place (paragraph and character,
// start and end) beside its quoted words; the words decide when the place no longer fits the text.

/** A highlight's place in one chapter: [paragraph, character] where it starts, and where it ends (exclusive). */
export type Span = { from: [number, number]; to: [number, number] };
/** A run of a paragraph's text, and the highlight it belongs to, if any. */
export type Run = { text: string; id: string | null };

const squash = (s: string) => s.replace(/\s+/g, " ").trim();

/** The words a span covers in a chapter's paragraphs, paragraphs joined by a space. */
export function spanText(paragraphs: readonly string[], s: Span): string {
  const parts: string[] = [];
  for (let i = s.from[0]; i <= s.to[0] && i < paragraphs.length; i++) {
    const p = paragraphs[i]!;
    parts.push(p.slice(i === s.from[0] ? s.from[1] : 0, i === s.to[0] ? s.to[1] : p.length));
  }
  return squash(parts.join(" "));
}

/**
 * Where a highlight sits in the chapter now: its kept place when the words there are still its quote, else the
 * first paragraph that holds the quote whole, else nowhere (the text changed too much to mark it).
 */
export function locate(paragraphs: readonly string[], s: Span, quote: string): Span | null {
  const q = squash(quote);
  if (!q) return null;
  if (s.from[0] < paragraphs.length && spanText(paragraphs, s) === q) return s;
  for (const [i, p] of paragraphs.entries()) {
    const at = p.indexOf(q);
    if (at >= 0) return { from: [i, at], to: [i, at + q.length] };
  }
  return null;
}

/** One paragraph cut into runs by the highlights that touch it; overlapping highlights go to the one kept first. */
export function runsOf(text: string, index: number, marks: readonly (Span & { id: string })[]): Run[] {
  const owner: (string | null)[] = new Array<string | null>(text.length).fill(null);
  for (const m of marks) {
    if (index < m.from[0] || index > m.to[0]) continue;
    const a = index === m.from[0] ? m.from[1] : 0;
    const b = index === m.to[0] ? m.to[1] : text.length;
    for (let k = Math.max(0, a); k < Math.min(text.length, b); k++) owner[k] ??= m.id;
  }
  const out: Run[] = [];
  for (let k = 0; k < text.length; k++) {
    const last = out.at(-1);
    if (last && last.id === owner[k]) last.text += text[k];
    else out.push({ text: text[k]!, id: owner[k]! });
  }
  return out.length ? out : [{ text, id: null }];
}

/** A quote as it is copied: the words in quotation marks, then who wrote them and where. */
export function citeQuote(quote: string, cite: { author?: string; title: string; place?: string }): string {
  return `“${squash(quote)}” — ${[cite.author, cite.title, cite.place].filter(Boolean).join(", ")}`;
}

/** A run with the character offset it starts at, as `cutRuns` gives it. */
export type PlacedRun = Run & { at: number };

/**
 * Runs split again at the given character offsets (where a note mark sits), each with the offset it starts at, so
 * a page can put something between two runs without changing the text.
 */
export function cutRuns(runs: readonly Run[], cuts: readonly number[]): PlacedRun[] {
  const sorted = [...new Set(cuts)].sort((a, b) => a - b);
  const out: PlacedRun[] = [];
  let at = 0;
  for (const r of runs) {
    let start = 0;
    for (const c of sorted) {
      const k = c - at;
      if (k <= start || k >= r.text.length) continue;
      out.push({ text: r.text.slice(start, k), id: r.id, at: at + start });
      start = k;
    }
    out.push({ text: r.text.slice(start), id: r.id, at: at + start });
    at += r.text.length;
  }
  return out;
}
