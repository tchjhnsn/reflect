import { describe, expect, it } from "vitest";
import { booksLike, workAccess, authorDetails, authorWorks, browseClass, browseQuery, isSort, searchAuthors, LANGUAGE, cleanEntry, coverUrl, searchBooks, trendingBooks, workDetails, workEntry, type Fetch } from "./openlibrary.ts";

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
    expect(seen[0]).toContain(`q=${encodeURIComponent("(lcc:T*) language:eng")}`);
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
    const trend: Fetch = async (url) => { seen.push(url); return { ok: true, status: 200, json: async () => ({ works: [{ key: "/works/OL9W", title: "Atomic Habits", author_name: ["James Clear"], cover_i: 12539702, language: ["eng"] }] }) }; };
    const [t] = await trendingBooks(trend, 5);
    expect(t).toMatchObject({ work: "OL9W", title: "Atomic Habits", cover: 12539702 });
    expect(seen[0]).toContain("/trending/weekly.json?limit=5");
    expect(coverUrl(t!)).toBe("https://covers.openlibrary.org/b/id/12539702-M.jpg");
    expect(coverUrl({})).toBeNull();
    const w = await workEntry("OL9W", fake([{ key: "/works/OL9W", title: "Atomic Habits", lcc: ["BF-0335"] }], seen));
    expect(w?.lcc).toEqual(["BF-0335"]);
    expect(seen[2]).toContain("q=key%3A%2Fworks%2FOL9W");
    expect(await workEntry("../etc", fake([], []))).toBeNull();
    expect(cleanEntry({ work: "OL9W", title: "x", cover: "12" })?.cover).toBe(12);
  });

  it("browses any query best rated first, with readers' ratings", async () => {
    const seen: string[] = [];
    const [b] = await browseQuery("lcc:[BF-0000 TO BF-9999.99999999]", fake([{ key: "/works/OL3W", title: "Mind", ratings_average: 4.123, ratings_count: 80 }], seen), { limit: 6 });
    expect(b!.rating).toEqual({ average: 4.12, count: 80 });
    expect(seen[0]).toContain("sort=rating");
    expect(seen[0]).toContain(encodeURIComponent("(lcc:[BF-0000"));
    expect(seen[0]).toContain(encodeURIComponent(") language:eng"));
    expect(seen[0]).toContain("&lang=en");
  });

  it("reads a work's description, rating and editions, and refuses a bad id", async () => {
    const seen: string[] = [];
    const routes: Record<string, unknown> = {
      "/works/OL9W.json": { description: { value: "A book about [habits](https://x).\n\n----------\nSee also" }, subjects: ["Habit", "Habit", "accessible book"] },
      "/works/OL9W/ratings.json": { summary: { average: 3.968, count: 1495 } },
      "/works/OL9W/editions.json?limit=30": { size: 42, entries: [{ title: "Atomic Habits", publish_date: "October 18, 2018", publishers: ["Avery"], isbn_13: ["9780735211292"], languages: [{ key: "/languages/eng" }] }, { publish_date: "1999" }] },
    };
    const f: Fetch = async (url) => {
      seen.push(url);
      const path = url.replace("https://openlibrary.org", "");
      return path in routes ? { ok: true, status: 200, json: async () => routes[path] } : { ok: false, status: 404, json: async () => ({}) };
    };
    const d = await workDetails("OL9W", f);
    expect(d).toEqual({
      description: "A book about habits.",
      subjects: ["Habit"],
      rating: { average: 3.97, count: 1495 },
      editions: { count: 42, first: [{ title: "Atomic Habits", year: 2018, publisher: "Avery", isbn: "9780735211292", language: "eng" }] },
      contents: [],
    });
    expect(await workDetails("OL0W", f)).toBeNull();
    expect(await workDetails("../x", f)).toBeNull();
  });

  it("lists books in English: English editions' titles and covers, an ISBN search in any language", async () => {
    expect(LANGUAGE.code).toBe("eng");
    const seen: string[] = [];
    const [b] = await searchBooks("alchemist", fake([{ key: "/works/OL5W", title: "O Alquimista", cover_i: 1, language: ["por", "eng"], editions: { docs: [{ title: "The Alchemist.", cover_i: 2, language: ["eng"] }] } }], seen));
    expect(b).toMatchObject({ title: "The Alchemist", cover: 2 });
    expect(seen[0]).toContain(encodeURIComponent("(alchemist) language:eng"));
    const [c] = await searchBooks("rich dad", fake([{ key: "/works/OL6W", title: "Rich Dad, Poor Dad", editions: { docs: [{ title: "Rich dad poor dad", language: ["eng"] }] } }], []));
    expect(c!.title).toBe("Rich Dad, Poor Dad"); // same title, only its case differs: the work's is kept
    await searchBooks("isbn:9780062315007", fake([], seen));
    expect(seen[1]).toContain("q=isbn%3A9780062315007&");
  });

  it("keeps this week's reads that have an English edition, shown by it", async () => {
    const seen: string[] = [];
    const f: Fetch = async (url) => {
      seen.push(url);
      if (url.includes("/trending/")) return { ok: true, status: 200, json: async () => ({ works: [{ key: "/works/OL7W", title: "O Alquimista", language: ["por", "eng"] }, { key: "/works/OL8W", title: "Только по-русски", language: ["rus"] }] }) };
      return { ok: true, status: 200, json: async () => ({ docs: [{ key: "/works/OL7W", title: "O Alquimista", editions: { docs: [{ title: "The Alchemist", language: ["eng"] }] } }] }) };
    };
    expect((await trendingBooks(f, 5)).map((b) => b.title)).toEqual(["The Alchemist"]);
    expect(seen[1]).toContain(encodeURIComponent('key:("/works/OL7W")'));
  });

  it("reads an author, their books and authors by name, and refuses a bad id", async () => {
    const seen: string[] = [];
    const routes: Record<string, unknown> = {
      "/authors/OL22060A.json": { name: "Augustine of Hippo", birth_date: "354", death_date: "430", bio: { value: "Bishop of [Hippo](https://x).\n\n----------\nsource" }, photos: [-1, 14343693], links: [{ url: "http://en.wikipedia.org/wiki/Augustine_of_Hippo" }] },
    };
    const f: Fetch = async (url) => {
      seen.push(url);
      const path = url.replace("https://openlibrary.org", "");
      if (path.startsWith("/search/authors.json")) return { ok: true, status: 200, json: async () => ({ docs: [{ key: "OL22060A", name: "Augustine of Hippo", work_count: 2291, top_work: "Confessions" }, { key: "OL1A", name: "Augustin X", work_count: 1 }, { key: "bad", name: "Bad" }] }) };
      if (path.startsWith("/search.json")) return { ok: true, status: 200, json: async () => ({ docs: [{ key: "/works/OL9W", title: "Confessions", author_name: ["Augustine of Hippo"], author_key: ["OL22060A", "nope"] }] }) };
      return path in routes ? { ok: true, status: 200, json: async () => routes[path] } : { ok: false, status: 404, json: async () => ({}) };
    };
    expect(await authorDetails("OL22060A", f)).toEqual({ id: "OL22060A", name: "Augustine of Hippo", birth: "354", death: "430", bio: "Bishop of Hippo.", photo: 14343693, wikipedia: "https://en.wikipedia.org/wiki/Augustine_of_Hippo" });
    expect(await authorDetails("OL0A", f)).toBeNull();
    expect(await authorDetails("../x", f)).toBeNull();
    const [w] = await authorWorks("OL22060A", f, { sort: "new", page: 2 });
    expect(w!.authorKeys).toEqual(["OL22060A"]);
    expect(seen.at(-1)).toContain(encodeURIComponent("(author_key:OL22060A) language:eng"));
    expect(seen.at(-1)).toContain("sort=new");
    expect(seen.at(-1)).toContain("page=2");
    expect(await authorWorks("x", f)).toEqual([]);
    expect((await searchAuthors("augustine", f)).map((a) => a.id)).toEqual(["OL22060A"]);
    expect(await searchAuthors("isbn:123", f)).toEqual([]);
    expect(isSort("old")).toBe(true);
    expect(isSort("toString")).toBe(false);
  });

  it("finds where a work is free to read or hear, and books like it", async () => {
    const seen: string[] = [];
    const f: Fetch = async (url) => {
      seen.push(url);
      if (url.includes("lending_identifier_s")) return { ok: true, status: 200, json: async () => ({ docs: [{ ebook_access: "public", lending_identifier_s: "prideprejudice00aust", lending_edition_s: "OL1M", id_project_gutenberg: ["1342", "x/y"], id_standard_ebooks: ["jane-austen/pride-and-prejudice", "../bad"], id_librivox: ["253", "orgullo"] }] }) };
      return { ok: true, status: 200, json: async () => ({ docs: [{ key: "/works/OL1W", title: "Self" }, { key: "/works/OL2W", title: "Emma" }] }) };
    };
    expect(await workAccess("OL1W", f)).toEqual({ access: "public", archive: "prideprejudice00aust", edition: "OL1M", gutenberg: ["1342"], standardEbooks: ["jane-austen/pride-and-prejudice"], librivox: ["253"] });
    expect(await workAccess("nope", f)).toBeNull();
    expect((await booksLike({ work: "OL1W", subjects: ["Fiction", "Courtship"] }, f)).map((b) => b.title)).toEqual(["Emma"]);
    expect(seen.at(-1)).toContain(encodeURIComponent('subject:"Courtship"'));
    expect(await booksLike({ work: "OL1W", subjects: ["Fiction"] }, f)).toEqual([]);
  });

  it("keeps editions' covers and pages and the first English table of contents", async () => {
    const routes: Record<string, unknown> = {
      "/works/OL9W.json": { description: "x" },
      "/works/OL9W/editions.json?limit=30": { size: 2, entries: [{ title: "Sin tapa", languages: [{ key: "/languages/spa" }], table_of_contents: [{ title: "Uno" }, { title: "Dos" }] }, { title: "With cover", covers: [-1, 77], number_of_pages: 320, languages: [{ key: "/languages/eng" }], table_of_contents: [{ title: "Part I", level: 0 }, { title: "Chapter 1", level: 1 }, "Notes"] }] },
    };
    const f: Fetch = async (url) => {
      const path = url.replace("https://openlibrary.org", "");
      return path in routes ? { ok: true, status: 200, json: async () => routes[path] } : { ok: false, status: 404, json: async () => ({}) };
    };
    const d = await workDetails("OL9W", f);
    expect(d?.editions.first[0]).toEqual({ title: "With cover", language: "eng", cover: 77, pages: 320 });
    expect(d?.contents).toEqual(["Part I", "  Chapter 1", "Notes"]);
  });
});
