// Claims in Learn: something a person can hold true or false ("marriage is a covenant"), apart from a
// subject they know. A claim names who holds it, each with a citation, as items do. The person's stance is
// one of the three doxastic attitudes of epistemology: belief, disbelief and suspension of judgment
// (Stanford Encyclopedia of Philosophy, "Belief"; Friedman, "Suspended Judgment", Philosophical Studies
// 162, 2013). Stances are never changed: each is a dated line, and the latest counts. A claim may sit under
// a subject. The stored shape follows schema.org Claim (text, appearance, author).

export const STANCES = [
  { id: "belief", text: "I believe it" },
  { id: "disbelief", text: "I disbelieve it" },
  { id: "suspension", text: "I suspend judgment" },
] as const;
export type Stance = (typeof STANCES)[number]["id"];
export const isStance = (s: string): s is Stance => STANCES.some((x) => x.id === s);

export type Holder = { name: string; citation?: string };
export type StanceLine = { stance: Stance; at: string; note?: string };

/** The stance that counts: the latest line, by time and then by order given. None until the person says. */
export function currentStance(lines: readonly StanceLine[]): StanceLine | null {
  let best: StanceLine | null = null;
  for (const l of lines) if (!best || l.at >= best.at) best = l;
  return best;
}

/** A claim's text and holders, cleaned. Throws on what the record cannot keep. */
export function cleanClaim(text: string, holders: readonly Holder[]): { text: string; holders: Holder[] } {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t) throw new Error("a claim needs words");
  if (t.length > 500) throw new Error("a claim is at most 500 characters");
  const hs = holders
    .map((h) => ({ name: h.name.replace(/\s+/g, " ").trim(), ...(h.citation?.trim() ? { citation: h.citation.trim() } : {}) }))
    .filter((h) => h.name);
  if (hs.length > 20) throw new Error("at most 20 holders on a claim");
  for (const h of hs) if (h.name.length > 200 || (h.citation?.length ?? 0) > 500) throw new Error("a holder's name is at most 200 characters and a citation at most 500");
  return { text: t, holders: hs };
}
