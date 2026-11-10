import { describe, expect, it } from "vitest";
import { asFeltThen, asOf, bindingProblems, presentMeaning, type Reading } from "./bindings.ts";

const text = "Called Dad tonight. I miss fishing at the lake with him.";

describe("bindings", () => {
  it("refuses a binding to nothing, an unknown element, or a phrase not in the entry", () => {
    expect(bindingProblems([{ element: "emotion", weight: 0.9, cues: [{ kind: "phrase", value: "fishing  at the lake" }] }], text)).toEqual([]);
    expect(bindingProblems([], text)).toEqual(["no bindings"]);
    expect(bindingProblems([{ element: "mood", weight: 2, cues: [] }], text)).toEqual(["not an element: mood", "weight out of range for mood", "mood is attached to nothing"]);
    expect(bindingProblems([{ element: "emotion", weight: 0.5, cues: [{ kind: "phrase", value: "the ocean" }] }], text)).toEqual(['not in the entry: "the ocean"']);
  });

  it("keeps the first reading as felt then, and answers as of a time", () => {
    const readings: Reading[] = [
      { readAt: "2026-06-01T00:00:00Z", bindings: [{ element: "emotion", weight: 0.3, cues: [{ kind: "place", value: "the lake" }] }] },
      { readAt: "2020-06-01T00:00:00Z", bindings: [{ element: "emotion", weight: 0.95, cues: [{ kind: "place", value: "the lake" }] }] },
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
