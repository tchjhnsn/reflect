import { describe, expect, it } from "vitest";
import { monthGrid, occurrences, shiftMonth, type CalEvent } from "./calendar";

const ev = (o: Partial<CalEvent>): CalEvent => ({ id: "e", tile: "work", title: "Shift", note: null, day: "2026-11-02", starts: null, ends: null, repeat: "none", until: null, recordId: null, ...o });

describe("the calendar", () => {
  it("places a one-off event and repeats weekly until its end", () => {
    expect(occurrences([ev({})], "2026-11-01", "2026-11-30").map((o) => o.day)).toEqual(["2026-11-02"]);
    expect(occurrences([ev({ repeat: "weekly", until: "2026-11-20" })], "2026-11-01", "2026-11-30").map((o) => o.day)).toEqual(["2026-11-02", "2026-11-09", "2026-11-16"]);
  });

  it("skips months without the day, and orders by day then time", () => {
    expect(occurrences([ev({ day: "2026-01-31", repeat: "monthly" })], "2026-01-01", "2026-05-31").map((o) => o.day)).toEqual(["2026-01-31", "2026-03-31", "2026-05-31"]);
    const list = occurrences([ev({ id: "b", title: "B", starts: "14:00" }), ev({ id: "a", title: "A", starts: "09:00" })], "2026-11-02", "2026-11-02");
    expect(list.map((o) => o.event.title)).toEqual(["A", "B"]);
  });

  it("lays out a month from the Monday before its first day", () => {
    const g = monthGrid("2026-11");
    expect(g[0]).toBe("2026-10-26");
    expect(g).toHaveLength(42);
    expect(shiftMonth("2026-12", 1)).toBe("2027-01");
  });
});
