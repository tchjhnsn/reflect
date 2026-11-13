import { describe, expect, it } from "vitest";
import { LISTS, bestListMatch, listBookQuery, listById, pathProgress } from "./lists.ts";

describe("lists", () => {
  it("keeps each list's source and books, with unique ids", () => {
    expect(new Set(LISTS.map((l) => l.id)).size).toBe(LISTS.length);
    for (const l of LISTS) {
      expect(l.source).toBeTruthy();
      expect(l.books.length).toBeGreaterThan(2);
    }
    expect(listById("harvard-classics")?.name).toBe("The Harvard Classics");
    expect(listById("nope")).toBeUndefined();
  });

  it("searches a book by title and, when named, author", () => {
    expect(listBookQuery({ title: "Utopia", author: "Thomas More" })).toBe('title:"Utopia" author:"Thomas More"');
    expect(listBookQuery({ title: "Beowulf", author: "" })).toBe('title:"Beowulf"');
  });

  it("says how far along a list read in order the person is", () => {
    expect(pathProgress(["read", "reading", null, "read"])).toEqual({ read: 2, next: 1 });
    expect(pathProgress(["read", "read"])).toEqual({ read: 2, next: null });
    expect(pathProgress([])).toEqual({ read: 0, next: null });
  });

  it("picks the book itself over a collection that holds it", () => {
    const r = (...t: string[]) => t.map((title) => ({ title }));
    const plato = (title: string) => ({ title, author: "Plato" });
    expect(bestListMatch(plato("Crito"), r("The trial and death of Socrates", "Platonis Apologia Socratis", "Platonis Crito"))?.title).toBe("Platonis Crito");
    expect(bestListMatch(plato("Phaedo"), r("The trial and death of Socrates", "Plato's Phaedo"))?.title).toBe("Plato's Phaedo");
    expect(bestListMatch(plato("Euthyphro"), r("Plato on the Trial", "Euthyphro", "The Euthyphro, Apology, & Crito"))?.title).toBe("Euthyphro");
    expect(bestListMatch(plato("Laws"), r("Something else"))?.title).toBe("Something else");
    expect(bestListMatch(plato("Laws"), [])).toBeNull();
  });
});
