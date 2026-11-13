import { describe, expect, it } from "vitest";
import { browseClass, cleanEntry, coverUrl, searchBooks, trendingBooks, workEntry, type Fetch } from "./openlibrary.ts";

const fake = (docs: unknown[], seen: string[]): Fetch => async (url) => {
  seen.push(url);
  return { ok: true, status: 200, json: async () => ({ docs }) };
};

describe("open library", () => {
  it("keeps a work's Library of Congress call numbers", async () => {
    const [b] = await searchBooks("chernobyl", fake([{ key: "/works/OL1W", title: "Midnight in Chernobyl", author_name: ["Adam Higginbotham"], lcc: ["TK-1362.00000000.U38"] }], []));
    expect(b!.lcc).toEqual(["TK-1362.00000000.U38"]);
    expect(b!.work).toBe("OL1W");
  });

  it("browses one main class, best rated first, and refuses anything but a class letter", async () => {
    const seen: string[] = [];
    const out = await browseClass("T", fake([{ key: "/works/OL2W", title: "Wiring" }, { title: "no key" }], seen), { page: 2, limit: 10 });
    expect(out.map((b) => b.title)).toEqual(["Wiring"]);
    expect(seen[0]).toContain("q=lcc%3AT*");
    expect(seen[0]).toContain("sort=rating");
    expect(seen[0]).toContain("page=2");
    await expect(browseClass("TK", fake([], []))).rejects.toThrow();
  });

  it("cleans an entry that came back from a browser, and refuses one with no work or title", () => {
    expect(cleanEntry({ work: "OL1W", title: " Wiring ", authors: ["A", 3, {}], year: "1999", lcc: ["TK-1"], contents: ["x"], extra: true })).toEqual({ work: "OL1W", title: "Wiring", authors: ["A", "3"], year: 1999, subjects: [], contents: [], lcc: ["TK-1"] });
    expect(cleanEntry({ title: "no work" })).toBeNull();
    expect(cleanEntry("nonsense")).toBeNull();
  });

  it("reads this week's trending works with their covers, and looks one work up by id", async () => {
    const seen: string[] = [];
    const trend: Fetch = async (url) => { seen.push(url); return { ok: true, status: 200, json: async () => ({ works: [{ key: "/works/OL9W", title: "Atomic Habits", author_name: ["James Clear"], cover_i: 12539702 }] }) }; };
    const [t] = await trendingBooks(trend, 5);
    expect(t).toMatchObject({ work: "OL9W", title: "Atomic Habits", cover: 12539702 });
    expect(seen[0]).toContain("/trending/weekly.json?limit=5");
    expect(coverUrl(t!)).toBe("https://covers.openlibrary.org/b/id/12539702-M.jpg");
    expect(coverUrl({})).toBeNull();
    const w = await workEntry("OL9W", fake([{ key: "/works/OL9W", title: "Atomic Habits", lcc: ["BF-0335"] }], seen));
    expect(w?.lcc).toEqual(["BF-0335"]);
    expect(seen[1]).toContain("q=key%3A%2Fworks%2FOL9W");
    expect(await workEntry("../etc", fake([], []))).toBeNull();
    expect(cleanEntry({ work: "OL9W", title: "x", cover: "12" })?.cover).toBe(12);
  });
});

