import { describe, expect, it } from "vitest";
import { homeCost, homeNeeds } from "./home";
import type { RecordFacts } from "./life";

describe("home", () => {
  const records: RecordFacts[] = [
    { kind: "food", state: null, body: { how: "self-provided", by: "kin", pantry: true } },
    { kind: "shelter", state: null, body: { how: "supplied", by: "market", cost: 900 } },
  ];
  const needs = homeNeeds(records);

  it("states each need with its standard in place and missing", () => {
    expect(needs.map((n) => n.id)).toEqual(["food", "water", "shelter", "transport", "safety"]);
    const food = needs.find((n) => n.id === "food")!;
    expect(food).toMatchObject({ stated: true, how: "Self-provided", by: "Kin", covered: true, done: 1, of: 1, missing: [] });
    expect(needs.find((n) => n.id === "water")).toMatchObject({ stated: false, done: 0, of: 2 });
  });

  it("sums what the person pays and names what others cover", () => {
    const c = homeCost(needs);
    expect(c.pay).toBe(900);
    expect(c.covered.map((n) => n.label)).toEqual(["Food"]);
    expect(homeCost(homeNeeds([])).pay).toBeNull();
  });
});
