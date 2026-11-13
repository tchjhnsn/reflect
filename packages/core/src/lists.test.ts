import { describe, expect, it } from "vitest";
import { LISTS, listBookQuery, listById } from "./lists.ts";

describe("lists", () => {
  it("keeps each list's source and books, with unique ids", () => {
    expect(new Set(LISTS.map((l) => l.id)).size).toBe(LISTS.length);
    for (const l of LISTS) {
      expect(l.source).toBeTruthy();
      expect(l.books.length).toBeGreaterThan(10);
    }
    expect(listById("harvard-classics")?.name).toBe("The Harvard Classics");
    expect(listById("nope")).toBeUndefined();
  });

  it("searches a book by title and, when named, author", () => {
    expect(listBookQuery({ title: "Utopia", author: "Thomas More" })).toBe('title:"Utopia" author:"Thomas More"');
    expect(listBookQuery({ title: "Beowulf", author: "" })).toBe('title:"Beowulf"');
  });
});
