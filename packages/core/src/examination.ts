/**
 * Putting yourself under examination (Torian, 2026-11-14): no grades and no model as examiner. A person says what
 * they will answer for from a course; people they invite examine them, on a call in Reflect or in person; each
 * examiner attests in their own words, once. The rules here are pure; the store keeps them.
 *
 * States of an examination: open (people can join and attest) -> closed (the record is kept as it stands).
 * An examiner: invited (has the link) -> joined -> attested (yes or not yet, with their words; never edited).
 */
export const EXAM = { name: 80, claim: 4000, attestation: 4000, examiners: 12 } as const;
export type ExamStatus = "open" | "closed";

export type JoinRefusal = "own" | "closed" | "full" | null;
/** Whether a person can join an examination as one of its examiners. */
export function joinRefusal(e: { candidate: string; status: ExamStatus; examiners: number }, person: string, already: boolean): JoinRefusal {
  if (e.candidate === person) return "own";
  if (already) return null;
  if (e.status !== "open") return "closed";
  return e.examiners >= EXAM.examiners ? "full" : null;
}

/** Cleans a name, a claim or an attestation: trimmed, within its length, never empty. */
export function cleanText(s: unknown, max: number): string | null {
  const t = typeof s === "string" ? s.replace(/\r\n/g, "\n").trim() : "";
  return t ? t.slice(0, max) : null;
}
