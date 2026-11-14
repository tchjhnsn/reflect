import { describe, expect, it } from "vitest";
import { batchOrder, daily, familyOf, filingsFrom, formingBatches, matches, parseRef, ranked, refHref, type Company } from "./work-browse";

const co = (name: string, batch: string): Company => ({ slug: name.toLowerCase(), name, line: "", about: "", batch, status: "Active", industries: ["B2B"], logo: null, site: null, team: null, where: null, launched: null });

describe("browsing work", () => {
  it("matches every word, any order, ignoring case and accents", () => {
    expect(matches("data scien", "Data Scientists")).toBe(true);
    expect(matches("café owner", "Cafe owners and managers")).toBe(true);
    expect(matches("nurse midwife", "Nurse Practitioners")).toBe(false);
    expect(matches("", "anything")).toBe(true);
  });

  it("ranks a name that starts with the query first", () => {
    const list = ["Chief Executives", "Executive Chefs", "Chefs and Head Cooks"];
    expect(ranked(list, "chef", (x) => x, () => [])).toEqual(["Chefs and Head Cooks", "Executive Chefs"]);
  });

  it("orders batches in time and finds the newest two as forming", () => {
    expect(batchOrder("Winter 2027")).toBeGreaterThan(batchOrder("Fall 2026"));
    expect(batchOrder("Unspecified")).toBe(0);
    expect(formingBatches([co("A", "Summer 2026"), co("B", "Winter 2027"), co("C", "Fall 2026"), co("D", "Unspecified")])).toEqual(["Winter 2027", "Fall 2026"]);
  });

  it("names a job's family from its code", () => {
    expect(familyOf("15-1252.00")?.label).toBe("Computers and mathematics");
    expect(familyOf("99-0000.00")).toBeNull();
  });

  it("keeps companies forming from SEC filings and leaves funds out", () => {
    const hits = [
      { _source: { display_names: ["Philo Film LLC  (CIK 0002165240)"], form: "D", file_date: "2026-11-13", biz_locations: ["Farmington, UT"] } },
      { _source: { display_names: ["Brenton Point Capital Partners 2005 Co-Invest Fund, LP  (CIK 0002165174)"], form: "D", file_date: "2026-11-13" } },
      { _source: { display_names: ["Philo Film LLC  (CIK 0002165240)"], form: "D", file_date: "2026-11-12" } },
      {},
    ];
    expect(filingsFrom(hits, "D")).toEqual([{ name: "Philo Film LLC", form: "D", day: "2026-11-13", where: "Farmington, UT" }]);
  });

  it("reads a reference to a catalog entry and links it", () => {
    expect(parseRef("job:15-1252.00")).toEqual({ kind: "job", id: "15-1252.00" });
    expect(parseRef("company:stripe")).toEqual({ kind: "company", id: "stripe" });
    expect(parseRef("job:../../etc")).toBeNull();
    expect(refHref({ kind: "fiction", id: "Q5446565" })).toBe("/life/work/imagined/Q5446565");
  });

  it("picks the same entries all day and others the next", () => {
    const list = Array.from({ length: 50 }, (_, i) => i);
    expect(daily(list, 5, "2026-11-14")).toEqual(daily(list, 5, "2026-11-14"));
    expect(daily(list, 5, "2026-11-14")).not.toEqual(daily(list, 5, "2026-11-15"));
    expect(new Set(daily(list, 50, "2026-11-14")).size).toBe(50);
  });
});
