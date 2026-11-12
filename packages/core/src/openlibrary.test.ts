import { describe, expect, it } from "vitest";
import { browseClass, cleanEntry, searchBooks, type Fetch } from "./openlibrary.ts";

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
});
