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
});
