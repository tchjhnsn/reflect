// Bindings: what each element of an episode was attached to (a time, a place, a person, a phrase, or a
// set of them), and how those attachments carry on across later episodes. Pure: the stores keep the
// records, this decides what they mean. Nothing here is stored; present meaning is computed when read.
import { PRIMES } from "./state.ts";

export type CueKind = "time" | "place" | "person" | "phrase";
export type Cue = { kind: CueKind; value: string };
/** One element, with a weight, attached to one or more cues. A set of cues is one attachment ("Dad + the lake"). */
export type Binding = { element: string; weight: number; cues: Cue[] };
/** The bindings kept together at one time: one reading of an entry. */
export type Reading = { readAt: string; bindings: Binding[] };
/** One binding of a cue, in one entry, at one reading. */
export type CuePoint = { entryId: string; localDate: string; element: string; weight: number; readAt: string };

const KINDS: readonly CueKind[] = ["time", "place", "person", "phrase"];

export const cleanCue = (c: Cue): Cue => ({ kind: c.kind, value: c.value.replace(/\s+/g, " ").trim() });

/** Problems with a reading's bindings, empty when it can be kept. A phrase must be the entry's own words. */
export function bindingProblems(bindings: readonly Binding[], entryText: string): string[] {
  const out: string[] = [];
  if (bindings.length === 0) out.push("no bindings");
  for (const b of bindings) {
    if (!(PRIMES as readonly string[]).includes(b.element)) out.push(`not an element: ${b.element}`);
    if (!(b.weight >= 0 && b.weight <= 1)) out.push(`weight out of range for ${b.element}`);
    if (b.cues.length === 0) out.push(`${b.element} is attached to nothing`);
    for (const c of b.cues.map(cleanCue)) {
      if (!KINDS.includes(c.kind)) out.push(`not a cue kind: ${c.kind}`);
      if (!c.value) out.push(`an empty ${c.kind} cue`);
      if (c.kind === "phrase" && c.value && !entryText.replace(/\s+/g, " ").includes(c.value)) out.push(`not in the entry: "${c.value}"`);
    }
  }
  return out;
}

/** The entry as felt then: its first reading. */
export const asFeltThen = (readings: readonly Reading[]): Reading | null =>
  [...readings].sort((a, b) => a.readAt.localeCompare(b.readAt))[0] ?? null;

/** The readings known at a time, oldest first. */
export const asOf = (readings: readonly Reading[], at: string): Reading[] =>
  readings.filter((r) => r.readAt <= at).sort((a, b) => a.readAt.localeCompare(b.readAt));

/**
 * How fast a binding fades, in days, by element. A parameter, not a finding: emotional bindings are taken
 * to fade more slowly than the rest (Yonelinas & Ritchey 2015, item-emotion versus item-context binding).
 */
export const HALF_LIFE_DAYS: Readonly<Record<string, number>> = { emotion: 1825, attachment: 1825 };
export const DEFAULT_HALF_LIFE_DAYS = 365;

/**
 * What a cue carries now, per element: `value` is the fade-weighted mean of every weight it was bound
 * with (later readings count more, earlier ones never drop out), and `strength` is how alive the
 * attachment is (the sum of the fade factors).
 */
export function presentMeaning(
  points: readonly CuePoint[],
  now: string,
  halfLives: Readonly<Record<string, number>> = HALF_LIFE_DAYS,
): Record<string, { value: number; strength: number }> {
  const acc: Record<string, { sum: number; strength: number }> = {};
  const t = Date.parse(now);
  for (const p of points) {
    const days = Math.max(0, (t - Date.parse(p.readAt)) / 86_400_000);
    const fade = 2 ** (-days / (halfLives[p.element] ?? DEFAULT_HALF_LIFE_DAYS));
    const a = (acc[p.element] ??= { sum: 0, strength: 0 });
    a.sum += p.weight * fade;
    a.strength += fade;
  }
  return Object.fromEntries(Object.entries(acc).map(([e, a]) => [e, { value: a.strength ? a.sum / a.strength : 0, strength: a.strength }]));
}
