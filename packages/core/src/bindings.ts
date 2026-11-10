// Bindings: what each element of an episode was attached to, and how those attachments carry on across
// later episodes. An element can attach to anything the entry names: a who or what (a noun), something that
// happened or was done (a verb), how something was (an adjective or adverb), a when, or a set of these
// ("Dad" + "left" + "quietly"). This follows appraisal theory, where feeling is a reaction to events, to
// agents' actions and to aspects of objects (Ortony, Clore & Collins 1988), and the event indexing model,
// where an episode is tracked by its people and things, time, place, causes and goals (Zwaan & Radvansky
// 1998). Pure: the stores keep the records, this decides what they mean. Present meaning is computed when read.
import { PRIMES } from "./state.ts";

/** What a concept is: a who or what (noun), a happening or act (verb), a quality (adjective, adverb), or a when. */
export type ConceptKind = "entity" | "event" | "quality" | "time";
/** What sort of entity: only an entity has one, and it is optional. */
export type EntitySort = "person" | "place" | "thing" | "group" | "idea";
/** A concept, kept once per person and found again by kind, sort and name. The name is the person's ("leave", "Dad"). */
export type Concept = { kind: ConceptKind; name: string; sort?: EntitySort };
/** The part a concept plays in what happened (thematic roles): "Dad left me" has Dad as agent and me as patient. */
export type Role = "agent" | "patient" | "place" | "time" | "manner" | "cause" | "goal";
/** One concept as this entry has it: the entry's own words ("left") and, when it matters, its role. */
export type Anchor = { concept: Concept; words?: string; role?: Role };
/** One element, with a weight, attached to one or more anchors. A set of anchors is one attachment. */
export type Binding = { element: string; weight: number; on: Anchor[] };
/** The bindings kept together at one time: one reading of an entry. */
export type Reading = { readAt: string; bindings: Binding[] };
/** One binding of a concept, in one entry, at one reading. */
export type ConceptPoint = { entryId: string; localDate: string; element: string; weight: number; readAt: string; role?: Role };

export const KINDS: readonly ConceptKind[] = ["entity", "event", "quality", "time"];
export const SORTS: readonly EntitySort[] = ["person", "place", "thing", "group", "idea"];
export const ROLES: readonly Role[] = ["agent", "patient", "place", "time", "manner", "cause", "goal"];

const squash = (s: string) => s.replace(/\s+/g, " ").trim();

export const cleanConcept = (c: Concept): Concept => ({ kind: c.kind, name: squash(c.name), ...(c.sort ? { sort: c.sort } : {}) });
export const cleanAnchor = (a: Anchor): Anchor => ({
  concept: cleanConcept(a.concept),
  ...(a.words !== undefined ? { words: squash(a.words) } : {}),
  ...(a.role ? { role: a.role } : {}),
});

/** The text a concept's identity is hashed from: the same kind, sort and name is the same concept. */
export const conceptKey = (c: Concept): string => {
  const k = cleanConcept(c);
  return `${k.kind}:${k.sort ?? ""}:${k.name.toLowerCase()}`;
};

/**
 * Problems with a reading's bindings, empty when it can be kept. Every anchor but a time must be the
 * entry's own words, so a concept is never attached to something the person did not write.
 */
export function bindingProblems(bindings: readonly Binding[], entryText: string): string[] {
  const out: string[] = [];
  const text = squash(entryText).toLowerCase();
  if (bindings.length === 0) out.push("no bindings");
  for (const b of bindings) {
    if (!(PRIMES as readonly string[]).includes(b.element)) out.push(`not an element: ${b.element}`);
    if (!(b.weight >= 0 && b.weight <= 1)) out.push(`weight out of range for ${b.element}`);
    if (b.on.length === 0) out.push(`${b.element} is attached to nothing`);
    for (const a of b.on.map(cleanAnchor)) {
      const c = a.concept;
      if (!KINDS.includes(c.kind)) out.push(`not a concept kind: ${c.kind}`);
      if (!c.name) out.push(`an unnamed ${c.kind}`);
      if (c.sort && (c.kind !== "entity" || !SORTS.includes(c.sort))) out.push(`not a sort for ${c.kind}: ${c.sort}`);
      if (a.role && !ROLES.includes(a.role)) out.push(`not a role: ${a.role}`);
      if (c.kind !== "time" && !a.words) out.push(`${c.name || c.kind} has no words from the entry`);
      if (a.words && !text.includes(a.words.toLowerCase())) out.push(`not in the entry: "${a.words}"`);
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
 * What a concept carries now, per element: `value` is the fade-weighted mean of every weight it was bound
 * with (later readings count more, earlier ones never drop out), and `strength` is how alive the
 * attachment is (the sum of the fade factors).
 */
export function presentMeaning(
  points: readonly ConceptPoint[],
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
