import { describe, expect, it } from "vitest";
import { bodySpace } from "./body";
import type { RecordFacts } from "./life";

describe("bodySpace", () => {
  const records: (RecordFacts & { id: string })[] = [
    { id: "h", kind: "health", state: null, body: { coverage: "Employer plan", sleep: 6.5, checkup: "2026-05-14" } },
    { id: "g", kind: "fitness", state: "in-progress", body: { goal: "Run a 5K", by: "2027-01-01" } },
    { id: "done", kind: "fitness", state: "reached", body: { goal: "Walk daily" } },
  ];
  const act = (kind: string, record: string, day: string, minutes: number) => ({ kind, record, day, minutes, amount: null });
  const acts = [act("vigorous", "g", "2026-11-13", 30), act("strength", "g", "2026-11-10", 20), act("moderate", "g", "2026-10-01", 60), act("slept", "h", "2026-11-13", 420), act("slept", "h", "2026-11-12", 390)];
  const s = bodySpace(records, acts, "2026-11-14");

  it("reads the week, sleep and the last checkup", () => {
    expect(s.week).toMatchObject({ minutes: 60, strengthDays: 1, met: false });
    expect(s.sleep).toEqual({ usual: 6.5, logged: 6.8, nights: 2 });
    expect(s.checkup).toEqual({ day: "2026-05-14", days: 184 });
    expect(s.health).toEqual({ coverage: "Employer plan", doctor: null });
  });

  it("keeps the goals still under way with four weeks of minutes toward each", () => {
    expect(s.goals).toEqual([{ id: "g", goal: "Run a 5K", by: "2027-01-01", state: "in-progress", minutes: 50, last: "2026-11-13" }]);
  });
});
