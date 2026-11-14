import { describe, expect, it } from "vitest";
import type { Body } from "./life";
import { friendsSpace, kinSpace, mateSpace, polisSpace } from "./tile-spaces";

const rec = (id: string, kind: string, state: string | null, body: Body) => ({ id, kind, state, body });
const act = (kind: string, record: string, day: string, minutes: number | null = null) => ({ kind, record, day, minutes, amount: null });
const today = "2026-11-14";

describe("tile spaces", () => {
  it("puts the friend gone quiet longest first", () => {
    const s = friendsSpace([rec("a", "friend", "close", { name: "Ann" }), rec("b", "friend", "friend", { name: "Ben" }), rec("c", "friend", "drifted", { name: "Cy" }), rec("n", "friend", "friend", { name: "New" }), rec("g", "group", "member", { name: "Choir" })],
      [act("saw", "a", "2026-11-10"), act("called", "b", "2026-09-01"), act("attended", "g", "2026-11-09")], today);
    expect(s.close.map((f) => [f.name, f.since])).toEqual([["New", null], ["Ben", 74], ["Ann", 4]]);
    expect(s.others.map((f) => f.name)).toEqual(["Cy"]);
    expect(s.groups).toEqual([{ id: "g", name: "Choir", state: "Member", last: "2026-11-09" }]);
  });

  it("splits dates planned from dates gone on", () => {
    const s = mateSpace([rec("s", "seeking", "seeking", { must: "Faith" }), rec("1", "date", "planned", { who: "R", day: "2026-11-20" }), rec("2", "date", "went", { who: "S", day: "2026-11-01", again: "yes" }), rec("3", "date", "planned", { who: "Old", day: "2026-10-01" })], today);
    expect(s.seeking).toMatchObject({ state: "Seeking", must: "Faith" });
    expect(s.planned.map((d) => d.who)).toEqual(["R"]);
    expect(s.went.map((d) => d.who)).toEqual(["S"]);
  });

  it("counts the care given as hours a week", () => {
    const s = kinSpace([rec("m", "relative", "some-help", { name: "Mom", born: "1950-01-01", carer: "me" }), rec("d", "relative", "independent", { name: "Dad" }), rec("x", "relative", "died", { name: "Gran" })],
      [act("cared", "m", "2026-11-10", 240), act("cared", "m", "2026-10-30", 240), act("cared", "m", "2026-09-01", 600), act("visited", "m", "2026-11-12")], today);
    expect(s.care).toMatchObject([{ name: "Mom", age: 76, careMinutes: 480, visited: "2026-11-12", carer: "me" }]);
    expect(s.others.map((k) => k.name)).toEqual(["Dad"]);
    expect(s.careHours).toBe(2);
  });

  it("orders elections and duties by day and flags what is late", () => {
    const s = polisSpace([rec("c", "civic", "voting", { residence: "Austin", registered: true }), rec("e", "election", "coming", { name: "General", day: "2026-11-30" }), rec("t", "duty", "due", { name: "Taxes", due: "2026-11-01" }), rec("r", "standing", "serving", { role: "Deacon", body: "Church" })], today);
    expect(s.civic).toMatchObject({ residence: "Austin", registered: true, state: "Voting" });
    expect(s.elections).toEqual([{ id: "e", name: "General", day: "2026-11-30", late: false }]);
    expect(s.duties[0]).toMatchObject({ name: "Taxes", late: true });
    expect(s.serving).toEqual([{ id: "r", role: "Deacon", where: "Church" }]);
  });
});
