// A draft reflection: the reader's guesses for one entry, written beside the person's records but never
// one of them. The person edits `chosen` and each term's `basis` and `keep`; `keepDraft` then turns what
// they left into a labeling, with every decision measured against the reader's guess.
import { GUESS_AT, PRIMES, decide, type Salience } from "./state.ts";
import type { LabelRow, SelfTermRow } from "./store.ts";

export type Draft = {
  entry: string;
  read_at: string;
  guess_at: number;
  weights: Record<string, number>;
  /** The elements the person says were part of the entry. Starts as the reader's guesses. */
  chosen: string[];
  self_terms: { quote: string; term: string; whose: "self" | "other"; model_basis: string; odds?: Record<string, number>; basis: string; keep: boolean }[];
};

export function makeDraft(entry: string, s: Salience, readAt: string): Draft {
  const guessAt = s.guessAt ?? GUESS_AT;
  return {
    entry,
    read_at: readAt,
    guess_at: guessAt,
    weights: s.weights,
    chosen: PRIMES.filter((p) => (s.weights[p] ?? 0) >= guessAt),
    self_terms: s.selfTerms.map((t) => ({ quote: t.quote, term: t.term, whose: t.whose, model_basis: t.basis, ...(t.odds ? { odds: t.odds } : {}), basis: t.basis, keep: true })),
  };
}

/** The labeling a draft keeps, or the reasons it cannot be kept. */
export function keepDraft(d: Draft): { labels: LabelRow[]; terms: SelfTermRow[] } {
  const unknown = d.chosen.filter((c) => !(PRIMES as readonly string[]).includes(c));
  if (unknown.length) throw new Error(`not elements: ${unknown.join(", ")}`);
  const labels = decide(d.weights, new Set(d.chosen), d.guess_at);
  const terms: SelfTermRow[] = d.self_terms.map((t) => {
    if (t.keep && !(PRIMES as readonly string[]).includes(t.basis)) throw new Error(`not an element: ${t.basis}`);
    return {
      quote: t.quote,
      term: t.term,
      whose: t.whose,
      modelBasis: t.model_basis,
      basis: t.keep ? t.basis : null,
      decision: !t.keep ? "rejected" : t.basis === t.model_basis ? "confirmed" : "corrected",
    };
  });
  return { labels, terms };
}
