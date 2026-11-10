// The state of the elements in a moment, mirrored from lean/NTS/State.lean via structure.json.
// A model may only propose salience weights and self-term bases; code checks them here, and the
// person confirms anything that is kept. Nothing in this module is a finding about the person.
import raw from "./structure.json";

type Structure = {
  primes: { id: string }[];
  introduction_order: string[];
  state: {
    arrangements: string[];
    amplitudes: string[];
    absorbs: { category: string; leader: string };
    categories: { id: string; pair: [string, string] }[];
    domains: { id: string; governor: string; proper: string[] }[];
  };
};

const S = raw as unknown as Structure;
export const PRIMES: readonly string[] = S.primes.map((p) => p.id);
export const INTRODUCTION: readonly string[] = S.introduction_order;
export const CATEGORIES = S.state.categories;
export const DOMAINS = S.state.domains;

export type Arrangement = "first_leads" | "second_leads" | "joined" | "opposed";
export type Amplitude = "low" | "raised" | "dominant" | "absorbed";
export type Region = { arrangement: Arrangement; amplitude: Amplitude };
export type Config = Record<string, Region>;

/** Where a moment sits in shared reality: the device's clock, and what the person states. */
export type Anchor = { clock: string; localDate: string; place?: string; bodyState?: string };
/** Temporality: the external clock and lived time as the person puts it. Both are kept. */
export type TimeLink = { anchor: Anchor; felt?: string };

/** Only the Apprehensions can be absorbed, and only with Numinous Sense leading. */
export function regionOk(category: string, r: Region): boolean {
  if (r.amplitude !== "absorbed") return true;
  const pair = CATEGORIES.find((c) => c.id === category)?.pair;
  return category === S.state.absorbs.category && pair?.[1] === S.state.absorbs.leader && r.arrangement === "second_leads";
}

/** A configuration names every category once, with no off state and absorption only where allowed. */
export function configOk(k: Config): boolean {
  return CATEGORIES.every((c) => k[c.id] !== undefined && regionOk(c.id, k[c.id]!)) && Object.keys(k).length === CATEGORIES.length;
}

/** While absorbed, every scope kept for the moment is held inside Numinous Sense. Scopes are paths, outermost first. */
export function scopesOk(k: Config, scopes: readonly (readonly string[])[]): boolean {
  if (k[S.state.absorbs.category]?.amplitude !== "absorbed") return true;
  return scopes.every((s) => s[0] === S.state.absorbs.leader);
}

/** A comparative term a person assigns. Its scope starts at Self-Differentiation; the next step is the basis. */
export type SelfTerm = { term: string; whose: "self" | "other"; comparedTo?: string; scope: readonly string[]; fact?: string };

export const basis = (t: SelfTerm): string | null => t.scope[1] ?? null;

export function selfTermOk(t: SelfTerm, maxDepth = 3): boolean {
  return t.scope[0] === "self-differentiation" && t.scope.length >= 2 && t.scope.length <= maxDepth && t.scope.every((p) => PRIMES.includes(p));
}

export function proper(domain: string): readonly string[] {
  const d = DOMAINS.find((x) => x.id === domain);
  if (!d) throw new Error(`unknown domain ${domain}`);
  return d.proper;
}

/** Silent unless the person asked; then only an open question, never a correction. */
export function onMismatch(asked: boolean, domain: string, basisPrime: string): { ask: string } | null {
  return asked && !proper(domain).includes(basisPrime) ? { ask: "What tells you that?" } : null;
}

/** A model's proposal: how prominent each element is in some words, and the bases of self-terms. */
export type Salience = {
  weights: Record<string, number>;
  /** The weight at or above which this reader's guess is shown; readers differ in how high they run. */
  guessAt?: number;
  /** `odds` are the reader's probabilities for each basis, when it gives them. */
  selfTerms: { quote: string; term: string; whose: "self" | "other"; basis: string; odds?: Record<string, number> }[];
};

/** Check a model's answer against the text. Returns null when the weights are unusable. */
export function parseSalience(text: string, answer: unknown): { salience: Salience | null; problems: string[] } {
  const problems: string[] = [];
  const w = (answer as { weights?: unknown })?.weights;
  if (typeof w !== "object" || w === null) return { salience: null, problems: ["no weights"] };
  const weights: Record<string, number> = {};
  for (const p of PRIMES) {
    const v = (w as Record<string, unknown>)[p];
    if (typeof v !== "number" || v < 0 || v > 1) problems.push(`weight for ${p} missing or outside 0..1`);
    else weights[p] = v;
  }
  const extra = Object.keys(w).filter((k) => !PRIMES.includes(k));
  if (extra.length) problems.push(`unknown elements: ${extra.join(", ")}`);
  if (Object.keys(weights).length !== PRIMES.length) return { salience: null, problems };
  const selfTerms: Salience["selfTerms"] = [];
  const list = (answer as { self_terms?: unknown }).self_terms;
  for (const t of Array.isArray(list) ? list : []) {
    const { quote, term, whose, basis: b } = (t ?? {}) as Record<string, unknown>;
    if (typeof quote !== "string" || !quote || !text.includes(quote)) problems.push(`self-term quote not in text: ${String(quote)}`);
    else if (whose !== "self" && whose !== "other") problems.push(`self-term whose must be self or other`);
    else if (typeof b !== "string" || !PRIMES.includes(b) || b === "self-differentiation") problems.push(`self-term basis is not another element: ${String(b)}`);
    else selfTerms.push({ quote, term: String(term ?? ""), whose, basis: b });
  }
  return { salience: { weights, selfTerms }, problems };
}

/** The element to ask about next: the most prominent not yet asked, ties in the introduction order (body first). */
export function nextFocus(s: Salience, asked: ReadonlySet<string>): string | null {
  let best: string | null = null;
  for (const p of INTRODUCTION) {
    if (asked.has(p)) continue;
    if (best === null || s.weights[p]! > s.weights[best]!) best = p;
  }
  return best;
}

const SPECIAL: Record<string, string> = { "self-differentiation": "Self-Differentiation", "aesthetic-sense": "Aesthetic Sense", "numinous-sense": "Numinous Sense" };
/** The element's display name: "aesthetic-sense" is "Aesthetic Sense". */
export const elementName = (id: string): string => SPECIAL[id] ?? id.charAt(0).toUpperCase() + id.slice(1);

/** A model weight at or above this is shown to the person as a guess to confirm or remove. */
export const GUESS_AT = 0.5;

export type LabelDecision = "confirmed" | "rejected" | "added" | "absent";

/** The person decides; the model's guess only sets what the question starts from. */
export function decide(weights: Record<string, number> | null, chosen: ReadonlySet<string>, guessAt = GUESS_AT): { element: string; weight: number | null; decision: LabelDecision }[] {
  return PRIMES.map((p) => {
    const w = weights?.[p] ?? null;
    const guessed = w !== null && w >= guessAt;
    const decision: LabelDecision = chosen.has(p) ? (guessed ? "confirmed" : "added") : guessed ? "rejected" : "absent";
    return { element: p, weight: w, decision };
  });
}
