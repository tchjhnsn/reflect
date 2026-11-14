import { describe, expect, it } from "vitest";
import { actText } from "./log";
import { APPS, checksOf, lifeLogActs, cleanBody, fitnessWeek, incomeSummary, kindOf, LIFE_APPS, lifeTone, TILE_KIND, titleOf, yearly } from "./life";

describe("life apps", () => {
  it("defines every app once, with unique kinds and well-formed fields", () => {
    expect(APPS.map((a) => a.id)).toEqual([...LIFE_APPS]);
    for (const a of APPS) {
      expect(new Set(a.kinds.map((k) => k.id)).size).toBe(a.kinds.length);
      for (const k of a.kinds) {
        expect(k.fields.some((f) => f.key === k.title)).toBe(true);
        expect(new Set(k.fields.map((f) => f.key)).size).toBe(k.fields.length);
        for (const f of k.fields) if (f.kind === "choice") expect(f.options?.length).toBeGreaterThan(1);
      }
    }
  });

  it("sends every unbuilt tile to an app and kind that exist", () => {
    for (const [tile, t] of Object.entries(TILE_KIND)) {
      if (t.kind) expect(kindOf(t.app, t.kind), tile).not.toBeNull();
    }
    expect(Object.keys(TILE_KIND).sort()).toEqual(["fitness", "food", "friends", "health", "income", "kin", "legacy", "mate", "polis", "safety", "shelter", "standing", "transport", "water", "work"]);
  });

  it("keeps only a kind's fields, each cleaned to its kind", () => {
    const k = kindOf("work", "income")!;
    expect(cleanBody(k, { source: "  Shop ", amount: "$1,200.50", per: "month", regular: "nonsense", extra: "x", hours: "-3" })).toEqual({ source: "Shop", amount: 1200.5, per: "month" });
    const food = kindOf("home", "food")!;
    expect(cleanBody(food, { pantry: "on", how: "stocked", cost: "abc" })).toEqual({ pantry: true, how: "stocked" });
    expect(cleanBody(kindOf("mate", "date")!, { day: "2026-13-40", who: "A" })).toEqual({ who: "A" });
  });

  it("names a record by its title field, a choice by its label", () => {
    expect(titleOf(kindOf("legacy", "plan")!, { name: "life-insurance" })).toBe("Life insurance");
    expect(titleOf(kindOf("friends", "friend")!, {})).toBe("Untitled friend");
    expect(titleOf(kindOf("work", "household")!, {})).toBe("Intended household");
    expect(titleOf(kindOf("home", "food")!, { how: "stocked" })).toBe("Food");
  });

  it("measures income coming in against the intended household", () => {
    expect(yearly({ amount: 16, per: "hour", hours: 40 })).toBe(33280);
    expect(yearly({ amount: 16, per: "hour" })).toBeNull();
    expect(yearly({ amount: 100, per: "once" })).toBe(0);
    const s = incomeSummary([
      { kind: "income", state: "coming-in", body: { amount: 16, per: "hour", hours: 40 } },
      { kind: "income", state: "coming-in", body: { amount: 200, per: "month" } },
      { kind: "income", state: "ended", body: { amount: 5000, per: "month" } },
      { kind: "income", state: null, body: { amount: 20, per: "hour" } },
      { kind: "household", state: null, body: { need: 71360 } },
    ]);
    expect(s).toEqual({ year: 35680, unknown: 1, need: 71360, share: 50 });
    expect(incomeSummary([]).share).toBeNull();
  });

  it("counts the week's activity against the guidelines, vigorous twice", () => {
    const acts = [
      { kind: "moderate", record: "g", day: "2026-11-14", minutes: 30, amount: null },
      { kind: "vigorous", record: "g", day: "2026-11-10", minutes: 30, amount: null },
      { kind: "moderate", record: "g", day: "2026-11-07", minutes: 300, amount: null },
      { kind: "strength", record: "g", day: "2026-11-12", minutes: 20, amount: null },
      { kind: "strength", record: "g", day: "2026-11-12", minutes: 20, amount: null },
      { kind: "strength", record: "g", day: "2026-11-13", minutes: 20, amount: null },
    ];
    expect(fitnessWeek(acts, "2026-11-14")).toEqual({ minutes: 90, strengthDays: 2, met: false });
    expect(fitnessWeek([...acts, { kind: "moderate", record: "g", day: "2026-11-11", minutes: 60, amount: null }], "2026-11-14").met).toBe(true);
  });

  it("gives a tile its tone from the person's records and recent acts", () => {
    const recs = [{ id: "a", kind: "friend", state: "drifted", body: { name: "Sam" } }];
    expect(lifeTone("friends", undefined, [], [], "2026-11-14")).toBe("unstated");
    expect(lifeTone("friends", undefined, recs, [], "2026-11-14")).toBe("thought");
    expect(lifeTone("friends", undefined, recs, [{ kind: "saw", record: "a", day: "2026-11-01", minutes: null, amount: null }], "2026-11-14")).toBe("action");
    expect(lifeTone("friends", undefined, recs, [{ kind: "saw", record: "a", day: "2026-09-01", minutes: null, amount: null }], "2026-11-14")).toBe("thought");
    expect(lifeTone("friends", undefined, [{ ...recs[0]!, state: "close" }], [], "2026-11-14")).toBe("action");
    expect(lifeTone("work", "income", [{ id: "w", kind: "work", state: "working", body: {} }], [], "2026-11-14")).toBe("unstated");
  });

  it("counts a need's checks", () => {
    const k = kindOf("home", "safety")!;
    expect(checksOf(k, { kit: true, fund: true })).toEqual({ done: 2, of: 4 });
  });

  it("gives the log each record kept, each later state and each act", () => {
    const recs = [{ id: "a", app: "friends" as const, kind: "friend", state: "close", body: { name: "Sam" }, createdAt: "2026-11-01T10:00:00.000Z" }];
    const acts = lifeLogActs(recs, [{ record: "a", state: "acquaintance", at: "2026-11-01T10:00:00.000Z" }, { record: "a", state: "close", at: "2026-11-05T10:00:00.000Z" }], [{ kind: "saw", record: "a", day: "2026-11-06", minutes: null, amount: null }]);
    expect(acts.map(actText)).toEqual(["Kept Sam in Friends", "Sam: Close friend", "Spent time with Sam"]);
  });
});
