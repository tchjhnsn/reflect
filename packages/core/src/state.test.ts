import { describe, expect, it } from "vitest";
import { CATEGORIES, DOMAINS, PRIMES, basis, configOk, nextFocus, onMismatch, parseSalience, proper, scopesOk, selfTermOk, type Config } from "./state.ts";

const calm: Config = Object.fromEntries(CATEGORIES.map((c) => [c.id, { arrangement: "joined", amplitude: "low" }]));
const flat = Object.fromEntries(PRIMES.map((p) => [p, 0.2]));

describe("element state", () => {
  it("has eight prime categories of two, and only Numinous absorbs", () => {
    expect(CATEGORIES).toHaveLength(8);
    expect(configOk(calm)).toBe(true);
    expect(configOk({ ...calm, drives: { arrangement: "first_leads", amplitude: "absorbed" } })).toBe(false);
    expect(configOk({ ...calm, apprehensions: { arrangement: "first_leads", amplitude: "absorbed" } })).toBe(false);
    const awe: Config = { ...calm, apprehensions: { arrangement: "second_leads", amplitude: "absorbed" } };
    expect(configOk(awe)).toBe(true);
    expect(scopesOk(awe, [["numinous-sense", "attention"]])).toBe(true);
    expect(scopesOk(awe, [["attention"]])).toBe(false);
    const { drives: _, ...missing } = calm;
    expect(configOk(missing)).toBe(false);
  });

  it("self-terms run through a basis under Self-Differentiation", () => {
    const tall = { term: "tall", whose: "self" as const, scope: ["self-differentiation", "logos", "reason"], fact: "height" };
    expect(selfTermOk(tall)).toBe(true);
    expect(basis(tall)).toBe("logos");
    expect(selfTermOk({ ...tall, scope: ["self-differentiation"] })).toBe(false);
    expect(selfTermOk({ ...tall, scope: ["emotion", "logos"] })).toBe(false);
  });

  it("Apprehensions are proper everywhere; a mismatch is silent unless asked, then only asks", () => {
    for (const d of DOMAINS) expect(proper(d.id)).toEqual(expect.arrayContaining(["aesthetic-sense", "numinous-sense", d.governor]));
    expect(proper("matter_of_fact")[0]).toBe("reason");
    expect(onMismatch(false, "matter_of_fact", "emotion")).toBeNull();
    expect(onMismatch(true, "matter_of_fact", "emotion")).toEqual({ ask: "What tells you that?" });
    expect(onMismatch(true, "matter_of_fact", "aesthetic-sense")).toBeNull();
  });

  it("checks a model's salience and picks the next focus deterministically", () => {
    const text = "Everyone calls me the big one.";
    const { salience, problems } = parseSalience(text, {
      weights: { ...flat, attachment: 0.8, emotion: 0.8 },
      self_terms: [
        { quote: "the big one", term: "big", whose: "self", basis: "logos" },
        { quote: "invented", term: "x", whose: "self", basis: "logos" },
      ],
    });
    expect(salience?.selfTerms.map((t) => t.term)).toEqual(["big"]);
    expect(problems).toHaveLength(1);
    expect(nextFocus(salience!, new Set())).toBe("attachment");
    expect(nextFocus(salience!, new Set(["attachment"]))).toBe("emotion");
    expect(nextFocus({ weights: flat, selfTerms: [] }, new Set())).toBe("embodiment");
    expect(parseSalience(text, { weights: { will: 2 } }).salience).toBeNull();
  });
});
