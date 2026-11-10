import { describe, expect, it } from "vitest";
import { asFeltThen, asOf, bindingProblems, conceptKey, presentMeaning, type Reading } from "./bindings.ts";

const text = "Dad left quietly tonight. I miss fishing at the lake with him.";

describe("bindings", () => {
  it("attaches an element to nouns, verbs and qualities, each anchored in the entry's words", () => {
    expect(bindingProblems([{ element: "emotion", weight: 0.9, on: [
      { concept: { kind: "entity", name: "Dad", sort: "person" }, words: "Dad", role: "agent" },
      { concept: { kind: "event", name: "leave" }, words: "left" },
      { concept: { kind: "quality", name: "quiet" }, words: "quietly", role: "manner" },
      { concept: { kind: "time", name: "night" } },
    ] }], text)).toEqual([]);
  });

  it("refuses a binding to nothing, an unknown element, words not in the entry, or a sort on a verb", () => {
    expect(bindingProblems([], text)).toEqual(["no bindings"]);
    expect(bindingProblems([{ element: "mood", weight: 2, on: [] }], text)).toEqual(["not an element: mood", "weight out of range for mood", "mood is attached to nothing"]);
    expect(bindingProblems([{ element: "emotion", weight: 0.5, on: [{ concept: { kind: "entity", name: "ocean", sort: "place" }, words: "the ocean" }] }], text)).toEqual(['not in the entry: "the ocean"']);
    expect(bindingProblems([{ element: "emotion", weight: 0.5, on: [{ concept: { kind: "event", name: "fish", sort: "thing" } }] }], text)).toEqual(["not a sort for event: thing", "fish has no words from the entry"]);
  });

  it("knows a concept again by kind, sort and name, whatever the case or spacing", () => {
    expect(conceptKey({ kind: "entity", name: " The  Lake", sort: "place" })).toBe(conceptKey({ kind: "entity", name: "the lake", sort: "place" }));
    expect(conceptKey({ kind: "event", name: "leave" })).not.toBe(conceptKey({ kind: "quality", name: "leave" }));
  });

  it("keeps the first reading as felt then, and answers as of a time", () => {
    const lake = { concept: { kind: "entity" as const, name: "the lake" }, words: "the lake" };
    const readings: Reading[] = [
      { readAt: "2026-06-01T00:00:00Z", bindings: [{ element: "emotion", weight: 0.3, on: [lake] }] },
      { readAt: "2020-06-01T00:00:00Z", bindings: [{ element: "emotion", weight: 0.95, on: [lake] }] },
    ];
    expect(asFeltThen(readings)?.readAt).toBe("2020-06-01T00:00:00Z");
    expect(asOf(readings, "2025-01-01T00:00:00Z").map((r) => r.readAt)).toEqual(["2020-06-01T00:00:00Z"]);
    expect(asOf(readings, "2027-01-01T00:00:00Z")).toHaveLength(2);
  });

  it("lets a later reading count more without the earlier one dropping out", () => {
    const points = [
      { entryId: "a", localDate: "2020-06-01", element: "emotion", weight: 1, readAt: "2020-06-01T00:00:00Z" },
      { entryId: "b", localDate: "2026-06-01", element: "emotion", weight: 0, readAt: "2026-06-01T00:00:00Z" },
      { entryId: "a", localDate: "2020-06-01", element: "reason", weight: 1, readAt: "2020-06-01T00:00:00Z" },
    ];
    const m = presentMeaning(points, "2026-06-01T00:00:00Z");
    expect(m.emotion!.value).toBeGreaterThan(0.3);
    expect(m.emotion!.value).toBeLessThan(0.5);
    expect(m.emotion!.strength).toBeGreaterThan(1);
    expect(m.reason!.strength).toBeLessThan(0.02);
  });
});
