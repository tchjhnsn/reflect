import { describe, expect, it } from "vitest";
import { citeQuote, locate, runsOf, spanText } from "./excerpts.ts";

const ps = ["It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.", "However little known the feelings or views of such a man may be."];

describe("highlights", () => {
  it("reads the words a highlight covers, across paragraphs", () => {
    expect(spanText(ps, { from: [0, 0], to: [0, 39] })).toBe("It is a truth universally acknowledged,");
    expect(spanText(ps, { from: [0, 110], to: [1, 7] })).toBe("a wife. However");
  });

  it("finds a highlight at its place, or by its words when the text moved, or not at all", () => {
    const s = { from: [0, 8] as [number, number], to: [0, 13] as [number, number] };
    expect(locate(ps, s, "truth")).toEqual(s);
    expect(locate(ps, { from: [1, 0], to: [1, 5] }, "truth")).toEqual(s);
    expect(locate(ps, s, "not in the text")).toBeNull();
    expect(locate(ps, { from: [9, 0], to: [9, 4] }, "feelings")).toEqual({ from: [1, 25], to: [1, 33] });
  });

  it("cuts a paragraph into plain and highlighted runs, the first highlight winning an overlap", () => {
    const marks = [{ id: "a", from: [0, 8], to: [0, 13] }, { id: "b", from: [0, 10], to: [1, 7] }] as const;
    const runs = runsOf(ps[0]!, 0, marks.map((m) => ({ id: m.id, from: [...m.from] as [number, number], to: [...m.to] as [number, number] })));
    expect(runs.slice(0, 3)).toEqual([{ text: "It is a ", id: null }, { text: "truth", id: "a" }, { text: ps[0]!.slice(13), id: "b" }]);
    expect(runs.map((r) => r.text).join("")).toBe(ps[0]);
    expect(runsOf(ps[1]!, 1, [{ id: "b", from: [0, 10], to: [1, 7] }])[0]).toEqual({ text: "However", id: "b" });
    expect(runsOf("", 0, [])).toEqual([{ text: "", id: null }]);
  });

  it("copies a quote with who wrote it and where", () => {
    expect(citeQuote("  It is a truth ", { author: "Jane Austen", title: "Pride and Prejudice", place: "Chapter I" })).toBe("“It is a truth” — Jane Austen, Pride and Prejudice, Chapter I");
  });
});
