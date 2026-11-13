import { describe, expect, it } from "vitest";
import { categoryOf, categoryOfCall, readCall, shelveThema, showCall, THEMA, themaCategory, themaQuery, themaSection } from "./thema.ts";

describe("thema", () => {
  it("has Thema's 20 sections, each child inside its section", () => {
    expect(THEMA.map((s) => s.code).join("")).toBe("ACDFGJKLMNPQRSTUVWXY");
    for (const s of THEMA) for (const c of s.children) expect(c.code).toMatch(new RegExp(`^${s.code}[A-Z]$`));
    expect(themaCategory("jm")?.heading).toBe("Psychology");
    expect(themaSection("FM")?.heading).toBe("Fiction");
  });

  it("reads call numbers in both forms", () => {
    expect(readCall("BF637.S8 C55")).toEqual({ letters: "BF", number: 637 });
    expect(readCall("BF-0637.00000000.S8")).toEqual({ letters: "BF", number: 637 });
    expect(readCall("B--0383")).toEqual({ letters: "B", number: 383 });
    expect(readCall("IX-1")).toBeNull();
    expect(showCall("BF-0335.30000000.C54")).toBe("BF335.3 .C54");
    expect(showCall("B--0383.00000000")).toBe("B383");
    expect(showCall("PA-4279.00000000 R4 1969")).toBe("PA4279 .R4 1969");
    expect(showCall("BS2625.3 .M6")).toBe("BS2625.3 .M6");
  });

  it("puts a call number in the narrowest category that covers it", () => {
    expect(categoryOfCall("BF-0335")).toBe("JM"); // psychology
    expect(categoryOfCall("BF-0637.S8")).toBe("VS"); // success, self-help
    expect(categoryOfCall("BF-1411")).toBe("VX"); // occultism
    expect(categoryOfCall("HM-0851")).toBe("JH"); // sociology
    expect(categoryOfCall("HD-0058.8")).toBe("KJ"); // management, inside economics' HB-HD
    expect(categoryOfCall("HB-0171")).toBe("KC");
    expect(categoryOfCall("QA-0076.73")).toBe("UY");
    expect(categoryOfCall("QA-0300")).toBe("PB");
    expect(categoryOfCall("BS-2625.3")).toBe("QR");
    expect(categoryOfCall("B--0383")).toBe("QD");
    expect(categoryOfCall("PS-3545")).toBeNull(); // a novel's call number names its literature, not a category
  });

  it("files fiction by genre from its subjects, and children's books apart", () => {
    expect(categoryOf({ lcc: ["PS-3619.A"], subjects: ["Fantasy fiction", "Magic"] })).toBe("FM");
    expect(categoryOf({ lcc: ["PS-3618"], subjects: ["Science fiction", "Thrillers"] })).toBe("FL"); // first subject leads
    expect(categoryOf({ lcc: ["PR-6054"], subjects: ["Fiction", "Juvenile fiction"] })).toBe("FB"); // call number outweighs the tag
    expect(categoryOf({ lcc: ["PZ-0007"], subjects: ["Fantasy"] })).toBe("YF");
    expect(categoryOf({ subjects: ["Picture books"] })).toBe("YB");
    expect(categoryOf({ lcc: ["BF-0335"], subjects: ["Psychology"] })).toBe("JM");
    expect(categoryOf({ subjects: ["Sociology"] })).toBe("JH");
    expect(categoryOf({ subjects: ["Romance languages"] })).toBeNull(); // no fiction, so no genre from a loose word
    expect(categoryOf({})).toBeNull();
  });

  it("searches a category by its call numbers when it has them, by subjects otherwise, fiction without children's books", () => {
    expect(themaQuery("JM")).toBe("lcc:[BF-0000 TO BF-9999.99999999]");
    expect(themaQuery("VS")).toBe("lcc:[BF-0632 TO BF-0637.99999999]");
    expect(themaQuery("QD")).toBe("lcc:[B--0000 TO BD-9999.99999999] OR lcc:[BH-0000 TO BJ-9999.99999999]");
    expect(themaQuery("FM")).toBe('(subject:"fantasy" OR subject:"fantasy fiction") -subject:"juvenile fiction" -lcc:[PZ-0000 TO PZ-9999.99999999]');
    expect(themaQuery("Q")).toContain(" OR ");
    expect(themaQuery("ZZ")).toBeNull();
  });

  it("shelves books by section and subjects by their books, guessing nothing", () => {
    const books = [
      { id: "habits", title: "Habits", lcc: ["BF-0335"], subjects: ["mind"] },
      { id: "kings", title: "Kings", lcc: ["PS-3619"], catalogSubjects: ["Fantasy fiction"], subjects: [] },
      { id: "notes", title: "Notes", subjects: ["wiring"] },
    ];
    const s = shelveThema(books, [{ id: "mind", name: "The mind" }, { id: "wiring", name: "Wiring" }]);
    expect(s.shelves.map((x) => x.id)).toEqual(["F", "J"]);
    expect(s.shelves[0]!.books[0]!.category).toBe("FM");
    expect(s.shelves[1]!.subjects.map((x) => x.id)).toEqual(["mind"]);
    expect(s.unshelved.books.map((b) => b.id)).toEqual(["notes"]);
    expect(s.unshelved.subjects.map((b) => b.id)).toEqual(["wiring"]);
    expect(shelveThema([], [], true).shelves).toHaveLength(20);
  });
});
