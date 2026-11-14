// Origins (docs/design/REFLECT_DOMAIN_2026-11-12.md, "Origins", decided): where something in a life comes
// from, as Aristotle's seven causes of action (Rhetoric I.10, 1368b32-1369a7), in his grouping. Three are not
// from the person (chance, nature, compulsion) and four are (habit, reasoning, spirit, appetite). Beliefs,
// telos and plans fall under reasoning. An item may have several origins; the person marks which are true.
// Spirit was "anger" until 11-14 (Torian): anger is one passion of spirit (thumos), the part of the soul Plato
// sets apart from appetite and reason (Republic IV 439e-441a; De Anima III.10). Origins kept as "anger" read as
// spirit; nothing stored is changed.

export const ORIGINS = [
  { id: "chance", label: "Chance", from: "not-me" },
  { id: "nature", label: "Nature", from: "not-me" },
  { id: "compulsion", label: "Compulsion", from: "not-me" },
  { id: "habit", label: "Habit", from: "me" },
  { id: "reasoning", label: "Reasoning", from: "me" },
  { id: "spirit", label: "Spirit", from: "me" },
  { id: "appetite", label: "Appetite", from: "me" },
] as const;
export type Origin = (typeof ORIGINS)[number]["id"];
export const isOrigin = (o: unknown): o is Origin => ORIGINS.some((x) => x.id === o);

/** The origins as given, known ones only, once each, in Aristotle's order. */
export function cleanOrigins(given: readonly unknown[]): Origin[] {
  const ids = given.map((g) => (g === "anger" ? "spirit" : g));
  return ORIGINS.map((o) => o.id).filter((id) => ids.includes(id));
}

export const originLabel = (o: Origin) => ORIGINS.find((x) => x.id === o)!.label;
