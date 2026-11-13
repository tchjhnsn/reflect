import { describe, expect, it } from "vitest";
import { carryPlace, findSpans, minutesFor, readingTime, searchBook, timeLeft, wordsOf } from "./reading.ts";

const chapters = [
  { title: "I", paragraphs: ["It is a truth universally acknowledged.", "A Truth again, and truth."] },
  { title: "II", paragraphs: ["No match here."] },
];

describe("reading", () => {
  it("counts words and minutes", () => {
    expect(wordsOf(["one two  three", " four "])).toBe(4);
    expect(minutesFor(0)).toBe(0);
    expect(minutesFor(10)).toBe(1);
    expect(minutesFor(2380)).toBe(10);
    expect(readingTime(130)).toBe("2 h 10 min");
    expect(readingTime(120)).toBe("2 h");
    expect(readingTime(5)).toBe("5 min");
  });

  it("gives the time left in the chapter and the book", () => {
    expect(timeLeft([2380, 2380, 2380], 2, 0.5)).toEqual({ chapter: 5, book: 15 });
    expect(timeLeft([2380], 1, 1)).toEqual({ chapter: 0, book: 0 });
  });

  it("finds a phrase in reading order, ignoring case", () => {
    const { hits, total } = searchBook(chapters, " TRUTH ");
    expect(total).toBe(3);
    expect(hits.map((h) => [h.chapter, h.p, h.c, h.match])).toEqual([[1, 0, 8, "truth"], [1, 1, 2, "Truth"], [1, 1, 19, "truth"]]);
    expect(searchBook(chapters, "truth", 1)).toMatchObject({ total: 3, hits: [{ c: 8 }] });
    expect(searchBook(chapters, "t").total).toBe(0);
    expect(findSpans(chapters[0]!.paragraphs, "truth")).toEqual([{ from: [0, 8], to: [0, 13] }, { from: [1, 2], to: [1, 7] }, { from: [1, 19], to: [1, 24] }]);
  });

  it("carries a place to a text with another chapter count", () => {
    expect(carryPlace({ chapter: 4, chapters: 62, at: 0.5 }, 62)).toEqual({ chapter: 4, at: 0.5 });
    expect(carryPlace({ chapter: 31, chapters: 62, at: 0 }, 61)).toEqual({ chapter: 30, at: 0 });
    expect(carryPlace({ chapter: 62, chapters: 62, at: 1 }, 61)).toEqual({ chapter: 61, at: 0 });
    expect(carryPlace({ chapter: 9, chapters: 5, at: 0 }, 5)).toEqual({ chapter: 5, at: 0 });
  });
});
