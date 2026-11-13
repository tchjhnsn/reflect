import { describe, expect, it } from "vitest";
import { googleVolume } from "./googlebooks.ts";
import type { Fetch } from "./openlibrary.ts";

const fake = (items: unknown[], seen: string[]): Fetch => async (url) => {
  seen.push(url);
  return { ok: true, status: 200, json: async () => ({ items }) };
};

describe("google books", () => {
  it("asks nothing without a key", async () => {
    const seen: string[] = [];
    expect(await googleVolume({ title: "Atomic Habits" }, fake([], seen), undefined)).toBeNull();
    expect(seen).toEqual([]);
  });

  it("finds a volume by ISBN and keeps its rating, plain description and an https link", async () => {
    const seen: string[] = [];
    const v = await googleVolume({ title: "Atomic Habits", isbn: "978-0735211292" }, fake([{ volumeInfo: { title: "Atomic Habits", averageRating: 4.5, ratingsCount: 900, description: "<p>Tiny <b>changes</b></p>", categories: ["Self-Help"], infoLink: "http://books.google.com/x" } }], seen), "k");
    expect(v).toEqual({ rating: { average: 4.5, count: 900 }, description: "Tiny changes", categories: ["Self-Help"], link: "https://books.google.com/x" });
    expect(seen[0]).toContain("q=isbn%3A9780735211292");
  });

  it("by title, takes only a volume whose title matches", async () => {
    const v = await googleVolume({ title: "The Republic", authors: ["Plato"] }, fake([{ volumeInfo: { title: "Plato: A Life" } }, { volumeInfo: { title: "The Republic of Plato", averageRating: 4 } }], []), "k");
    expect(v?.rating).toBeNull();
    expect(await googleVolume({ title: "The Republic" }, fake([{ volumeInfo: { title: "Something else" } }], []), "k")).toBeNull();
  });

  it("takes the rating from the best-rated edition with the same title when the ISBN's edition has none", async () => {
    const seen: string[] = [];
    const f: Fetch = async (url) => {
      seen.push(url);
      const items = url.includes("isbn%3A")
        ? [{ volumeInfo: { title: "Atomic Habits", infoLink: "https://books.google.com/a" } }]
        : [{ volumeInfo: { title: "Atomic Habits", averageRating: 4, ratingsCount: 12 } }, { volumeInfo: { title: "Atomic Habits", averageRating: 4.5, ratingsCount: 300 } }, { volumeInfo: { title: "Other", averageRating: 5, ratingsCount: 9000 } }];
      return { ok: true, status: 200, json: async () => ({ items }) };
    };
    const v = await googleVolume({ title: "Atomic Habits", authors: ["James Clear"], isbn: "9780735211292" }, f, "k");
    expect(v).toMatchObject({ rating: { average: 4.5, count: 300 }, link: "https://books.google.com/a" });
    expect(seen).toHaveLength(2);
  });
});
