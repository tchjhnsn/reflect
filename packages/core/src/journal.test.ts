import { describe, expect, it } from "vitest";
import type { IsoDate } from "./person.ts";
import { CLOSED, OPENING, TELL_ME_MORE, acceptLocalDate, byDay, clean, nextPrompt, onThisDay, type Entry } from "./journal.ts";

const entry = (id: string, localDate: string, createdAt: string, parts = 1): Entry => ({
  id,
  localDate: localDate as IsoDate,
  createdAt,
  promptId: null,
  closedAt: null,
  parts: Array.from({ length: parts }, (_, i) => ({ seq: i, text: `p${i}`, createdAt })),
});

describe("journal", () => {
  it("asks the same fixed things, with no model", () => {
    expect(nextPrompt({ parts: [], closedAt: null })).toBe(OPENING);
    expect(nextPrompt({ parts: [{ seq: 0, text: "x", createdAt: "" }], closedAt: null })).toBe(TELL_ME_MORE);
    expect(nextPrompt({ parts: [{ seq: 0, text: "x", createdAt: "" }], closedAt: "t" })).toBe(CLOSED);
  });

  it("treats blank writing as nothing", () => {
    expect(clean("  \n\t ")).toBeNull();
    expect(clean("  rain at the park \n")).toBe("rain at the park");
  });

  it("finds this day in earlier years only", () => {
    const es = [entry("a", "2024-11-09", "2024-11-09T10:00Z"), entry("b", "2026-11-09", "2026-11-09T10:00Z"), entry("c", "2025-11-09", "2025-11-09T10:00Z"), entry("d", "2025-11-08", "2025-11-08T10:00Z")];
    expect(onThisDay(es, "2026-11-09" as IsoDate).map((e) => e.id)).toEqual(["c", "a"]);
  });

  it("groups by day, newest first", () => {
    const es = [entry("a", "2026-11-08", "2026-11-08T09:00Z"), entry("b", "2026-11-09", "2026-11-09T08:00Z"), entry("c", "2026-11-09", "2026-11-09T21:00Z")];
    expect(byDay(es).map((d) => [d.day, d.entries.map((e) => e.id)])).toEqual([
      ["2026-11-09", ["c", "b"]],
      ["2026-11-08", ["a"]],
    ]);
  });

  it("accepts a local date within a day of the server and refuses back-dating", () => {
    const now = new Date("2026-11-09T03:00:00Z");
    expect(acceptLocalDate("2026-11-08" as IsoDate, now)).toBe(true);
    expect(acceptLocalDate("2026-11-09" as IsoDate, now)).toBe(true);
    expect(acceptLocalDate("2026-11-01" as IsoDate, now)).toBe(false);
  });
});
