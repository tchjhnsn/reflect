import { describe, expect, it } from "vitest";
import { byMonth, calendar, streak } from "./habits.ts";

describe("reading habits", () => {
  it("counts the days in a row", () => {
    expect(streak(["2026-11-11", "2026-11-12", "2026-11-13"], "2026-11-13")).toBe(3);
    expect(streak(["2026-11-11", "2026-11-12"], "2026-11-13")).toBe(2);
    expect(streak(["2026-11-10"], "2026-11-13")).toBe(0);
    expect(streak(["2026-02-28", "2026-03-01"], "2026-03-01")).toBe(2);
  });

  it("lays the days out as weeks, Monday first", () => {
    const c = calendar(["2026-11-09", "2026-11-13"], "2026-11-13", 2);
    expect(c).toHaveLength(2);
    expect(c[0]![0]).toEqual({ day: "2026-11-02", read: false });
    expect(c[1]![0]).toEqual({ day: "2026-11-09", read: true });
    expect(c[1]![4]).toEqual({ day: "2026-11-13", read: true });
    expect(c[1]![5]).toBeNull();
  });

  it("groups a year's finished books by month", () => {
    const m = byMonth([{ finishedOn: "2026-03-09" }, { finishedOn: "2026-03-01" }, { finishedOn: "2025-03-01" }, { finishedOn: null }], 2026);
    expect([...m.entries()]).toEqual([[3, [{ finishedOn: "2026-03-01" }, { finishedOn: "2026-03-09" }]]]);
  });
});
