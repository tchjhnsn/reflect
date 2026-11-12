import { describe, expect, it } from "vitest";
import { classOf, LCC_CLASSES, mainClass, shelve } from "./shelves.ts";

describe("shelves", () => {
  it("has the 21 main classes of the Library of Congress Classification", () => {
    expect(LCC_CLASSES.map((c) => c.id).join("")).toBe("ABCDEFGHJKLMNPQRSTUVZ");
  });

  it("reads the main class from either call number form", () => {
    expect(classOf("BS2625.3 .M6")).toBe("B");
    expect(classOf("JC-0071.00000000.P35")).toBe("J");
    expect(classOf("Z--0473.00000000")).toBe("Z");
    expect(classOf("IX-1")).toBeNull();
    expect(classOf("")).toBeNull();
  });

  it("takes the class most call numbers point to, ties to the first", () => {
    expect(mainClass(["PA-4279", "JC-0071", "JC-0071.P3", "B--0383"])).toBe("J");
    expect(mainClass(["PA-1", "JC-1"])).toBe("P");
    expect(mainClass([])).toBeNull();
    expect(mainClass(undefined)).toBeNull();
  });

  it("shelves books by call number and subjects by their books, guessing nothing", () => {
    const books = [
      { id: "acts", title: "Acts", lcc: ["BS-2625.3"], subjects: ["bible"] },
      { id: "republic", title: "Republic", lcc: ["JC-0071", "PA-4279", "JC-0071.P35"], subjects: ["justice", "bible"] },
      { id: "romans", title: "Romans", lcc: ["BS-2665"], subjects: ["bible"] },
      { id: "notes", title: "Notes", subjects: ["wiring"] },
    ];
    const subjects = [
      { id: "bible", name: "The Bible" },
      { id: "justice", name: "Justice" },
      { id: "wiring", name: "Electrical wiring" },
    ];
    const s = shelve(books, subjects);
    expect(s.shelves.map((x) => x.id)).toEqual(["B", "J"]);
    expect(s.shelves[0]!.books.map((b) => b.id)).toEqual(["acts", "romans"]);
    expect(s.shelves[0]!.subjects.map((b) => b.id)).toEqual(["bible"]);
    expect(s.shelves[1]!.subjects.map((b) => b.id)).toEqual(["justice"]);
    expect(s.unshelved.books.map((b) => b.id)).toEqual(["notes"]);
    expect(s.unshelved.subjects.map((b) => b.id)).toEqual(["wiring"]);
  });

  it("keeps every shelf, empty ones too, when asked to browse all", () => {
    expect(shelve([], [], true).shelves).toHaveLength(21);
    expect(shelve([], []).shelves).toHaveLength(0);
  });
});
