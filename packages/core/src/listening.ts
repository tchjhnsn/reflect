// Reading and listening together: which recorded section of an audiobook reads which chapter of the text. Pure.

const ROMAN: Record<string, number> = { i: 1, v: 5, x: 10, l: 50, c: 100, d: 500, m: 1000 };

/** A Roman numeral's value, or null when the letters are not one. */
export function romanValue(s: string): number | null {
  const t = s.toLowerCase();
  if (!/^[ivxlcdm]+$/.test(t)) return null;
  let n = 0;
  for (let i = 0; i < t.length; i++) {
    const v = ROMAN[t[i]!]!;
    const next = ROMAN[t[i + 1] ?? ""] ?? 0;
    n += v < next ? -v : v;
  }
  return n > 0 && n < 4000 ? n : null;
}

const WORD = /\b(chapter|letter|book|part|canto|act|section|volume|stave|essay)\b\.?\s+([0-9]+|[ivxlcdm]+)\b/gi;

/**
 * What a chapter or section is called, as a key that both a text and a recording would agree on: each kind and
 * number in its title ("Chapter V" and "Chapter 05 - The Creation" are both "chapter 5"), or null with none.
 */
export function sectionKey(title: string): string | null {
  const parts: string[] = [];
  for (const m of title.matchAll(WORD)) {
    const n = /^\d+$/.test(m[2]!) ? Number(m[2]) : romanValue(m[2]!);
    if (n) parts.push(`${m[1]!.toLowerCase()} ${n}`);
  }
  return parts.length ? parts.join(" ") : null;
}

/**
 * For each chapter of the text (in order), the index of the recorded section that reads it, or null: by what each
 * is called, where a name is in both and only once, when that matches at least half the chapters; else, when the
 * counts agree, chapter by chapter; else by name as far as it goes.
 */
export function matchSections(chapters: readonly string[], sections: readonly string[]): (number | null)[] {
  const counts = (keys: (string | null)[]) => keys.reduce((m, k) => (k ? m.set(k, (m.get(k) ?? 0) + 1) : m), new Map<string, number>());
  const ck = chapters.map(sectionKey);
  const sk = sections.map(sectionKey);
  const cc = counts(ck);
  const sc = counts(sk);
  const byName = ck.map((k) => (k && cc.get(k) === 1 && sc.get(k) === 1 ? sk.indexOf(k) : null));
  const named = byName.filter((x) => x !== null).length;
  if (named * 2 >= chapters.length && named > 0) return byName;
  return chapters.length === sections.length ? chapters.map((_, i) => i) : byName;
}
