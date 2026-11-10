import { describe, expect, it } from "vitest";
import { PRIMES, decide } from "@/core/state";
import { JEV_GUESS_AT, foundTerms, fromJev } from "./reader";

const k = (p: string) => p.replaceAll("-", "_");
const nouls = (v: number) => Object.fromEntries(PRIMES.map((p) => [k(p), { type: "noul" as const, noul: v }]));
const text = "My brothers were the clever ones; I was the quiet one.";

describe("reader", () => {
  it("keeps only self-terms quoted from the text", () => {
    const found = foundTerms(text, {
      self_terms: [
        { quote: "the quiet one", term: "quiet", whose: "self" },
        { quote: "the loud one", term: "loud", whose: "self" },
        { quote: "the clever ones", term: "", whose: "other" },
        { quote: "the quiet one", term: "quiet", whose: "self" },
        { quote: "My brothers", term: "x", whose: "nobody" },
      ],
    });
    expect(found).toEqual([
      { quote: "the quiet one", term: "quiet", whose: "self" },
      { quote: "the clever ones", term: "the clever ones", whose: "other" },
    ]);
    expect(foundTerms(text, null)).toEqual([]);
  });

  it("reads Jev's answers as weights and bases with odds", () => {
    const answers = {
      ...nouls(0.4),
      emotion: { type: "noul" as const, noul: 0.97 },
      basis_0: { type: "choice" as const, choice: "emotion", probabilities: { emotion: 0.6, will: 0.3, self_differentiation: 0.1 } },
    };
    const s = fromJev(answers, [{ quote: "the quiet one", term: "quiet", whose: "self" }])!;
    expect(s.weights.emotion).toBe(0.97);
    expect(s.guessAt).toBe(JEV_GUESS_AT);
    expect(s.selfTerms).toEqual([{ quote: "the quiet one", term: "quiet", whose: "self", basis: "emotion", odds: { emotion: 0.6, will: 0.3 } }]);
  });

  it("gives no reading when an element is unanswered, and drops a basis outside the elements", () => {
    const { emotion: _e, ...missing } = nouls(0.5);
    expect(fromJev(missing, [])).toBeNull();
    const s = fromJev({ ...nouls(0.5), basis_0: { type: "choice", choice: "self_differentiation" } }, [{ quote: "q", term: "q", whose: "self" }])!;
    expect(s.selfTerms).toEqual([]);
  });

  it("counts a guess against the reader's own threshold", () => {
    const weights = Object.fromEntries(PRIMES.map((p) => [p, 0.8]));
    expect(decide(weights, new Set(), 0.5).find((r) => r.element === "will")!.decision).toBe("rejected");
    expect(decide(weights, new Set(), JEV_GUESS_AT).find((r) => r.element === "will")!.decision).toBe("absent");
  });
});
