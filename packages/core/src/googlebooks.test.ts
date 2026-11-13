import { describe, expect, it } from "vitest";
import { googleVolume, readerUrl, sameBook } from "./googlebooks.ts";
import type { Fetch } from "./openlibrary.ts";

const fake = (items: unknown[], seen: string[]): Fetch => async (url) => {
  seen.push(url);
  return { ok: true, status: 200, json: async () => ({ items }) };
};
const preview = { viewability: "PARTIAL", embeddable: true };
const whole = { viewability: "ALL_PAGES", embeddable: true };

describe("google books", () => {
  it("asks nothing without a key", async () => {
    const seen: string[] = [];
    expect(await googleVolume({ title: "Atomic Habits" }, fake([], seen), undefined)).toBeNull();
    expect(seen).toEqual([]);
  });

  it("finds a volume by ISBN and keeps its plain description and an https link, and no rating", async () => {
    const seen: string[] = [];
    const v = await googleVolume({ title: "Atomic Habits", isbn: "978-0735211292" }, fake([{ id: "abc", volumeInfo: { title: "Atomic Habits", averageRating: 4.5, ratingsCount: 900, description: "<p>Tiny <b>changes</b></p>", categories: ["Self-Help"], infoLink: "http://books.google.com/x" }, accessInfo: preview }], seen), "k");
    expect(v).toEqual({ read: { id: "abc", full: false }, description: "Tiny changes", categories: ["Self-Help"], link: "https://books.google.com/x" });
    expect(seen[0]).toContain("q=isbn%3A9780735211292");
    expect(seen[0]).toContain("country=US");
  });

  it("knows the same book from a workbook, a summary or a book about it", () => {
    const habits = { title: "Atomic Habits", authors: ["James Clear"] };
    expect(sameBook(habits, { title: "Atomic Habits", authors: ["James Clear"] })).toBe(true);
    expect(sameBook(habits, { title: "The Atomic Habits Workbook", authors: ["James Clear"] })).toBe(false);
    expect(sameBook(habits, { title: "Atomic Habits and James Clear", authors: ["Someone Else"] })).toBe(false);
    expect(sameBook(habits, { title: "Atomic Habits", authors: ["Summary Press"] })).toBe(false);
    expect(sameBook({ title: "Confessions", authors: ["Augustine of Hippo"] }, { title: "The Confessions of S. Augustin", authors: ["Saint Augustine (Bishop of Hippo.)"] })).toBe(true);
    expect(sameBook({ title: "The Republic" }, { title: "Republic" })).toBe(true);
    expect(sameBook({ title: "The Republic" }, { title: "Plato: A Life" })).toBe(false);
  });

  it("reads from the English edition of the same book with the most to read: the whole book first, then a preview", async () => {
    const seen: string[] = [];
    const f: Fetch = async (url) => {
      seen.push(url);
      const items = url.includes("isbn%3A")
        ? [{ id: "isbn1", volumeInfo: { title: "Confessions", authors: ["Augustine"], infoLink: "https://books.google.com/a" }, accessInfo: { viewability: "NO_PAGES" } }]
        : [
            { id: "p1", volumeInfo: { title: "The Confessions", authors: ["Saint Augustine"], language: "en" }, accessInfo: preview },
            { id: "fr", volumeInfo: { title: "Confessions", authors: ["Augustin"], language: "fr" }, accessInfo: whole },
            { id: "noembed", volumeInfo: { title: "Confessions", authors: ["Augustine"], language: "en" }, accessInfo: { viewability: "ALL_PAGES", embeddable: false } },
            { id: "w1", volumeInfo: { title: "The Confessions of S. Augustin", authors: ["Saint Augustine"], language: "en" }, accessInfo: whole },
            { id: "other", volumeInfo: { title: "Augustine's Confessions: A Guide", authors: ["Augustine Scholar"], language: "en" }, accessInfo: whole },
          ];
      return { ok: true, status: 200, json: async () => ({ items }) };
    };
    const v = await googleVolume({ title: "Confessions", authors: ["Augustine"], isbn: "9780140441147" }, f, "k");
    expect(v).toMatchObject({ read: { id: "w1", full: true }, link: "https://books.google.com/a" });
    expect(seen).toHaveLength(2);
  });

  it("has no reader when no edition can be read, and nothing when no volume is the same book", async () => {
    expect((await googleVolume({ title: "Atomic Habits" }, fake([{ id: "x", volumeInfo: { title: "Atomic Habits" }, accessInfo: { viewability: "NO_PAGES" } }], []), "k"))?.read).toBeNull();
    expect(await googleVolume({ title: "The Republic" }, fake([{ volumeInfo: { title: "Something else" } }], []), "k")).toBeNull();
  });

  it("frames Google's reader at the start of the book", () => {
    expect(readerUrl("w1")).toBe("https://books.google.com/books?id=w1&pg=PP1&output=embed");
  });
});
