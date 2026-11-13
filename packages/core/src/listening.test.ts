import { describe, expect, it } from "vitest";
import { matchSections, romanValue, sectionKey } from "./listening.ts";

describe("listening along", () => {
  it("reads Roman numerals and section names", () => {
    expect([romanValue("XIV"), romanValue("mcmxc"), romanValue("abc")]).toEqual([14, 1990, null]);
    expect(sectionKey("Chapter V")).toBe("chapter 5");
    expect(sectionKey("Chapter 05 - The Creation")).toBe("chapter 5");
    expect(sectionKey("Book II: Chapter III")).toBe("book 2 chapter 3");
    expect(sectionKey("Preface")).toBeNull();
  });

  it("matches chapters to recorded sections", () => {
    expect(matchSections(["I", "II"], ["One", "Two"])).toEqual([0, 1]);
    expect(matchSections(["Letter 1", "Chapter I", "Chapter II", "Notes"], ["Preface", "Letter 01", "Chapter 01", "Chapter 02"])).toEqual([1, 2, 3, null]);
    expect(matchSections(["Chapter I", "Chapter I"], ["Chapter 1", "x", "y"])).toEqual([null, null]);
  });
});
