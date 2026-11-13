import { describe, expect, it } from "vitest";
import { bookOf, hardcoverAuth, hardcoverBook, mostCommon, pickSearchHit, translatorsOf } from "./hardcover.ts";

const row = {
  id: 7, slug: "the-iliad", title: "The Iliad", rating: "4.123", ratings_count: 900, users_count: 5000,
  editions: [
    { id: 1, title: "Ilias", users_count: 900, language: { language: "German" } },
    { id: 2, title: "The Iliad", isbn_13: "9780140275360", release_year: 1998, pages: 683, edition_format: "Paperback", users_count: 800, rating: 4.21, publisher: { name: "Penguin" }, language: { language: "English" }, cached_contributors: [{ author: { name: "Homer" } }, { author: { name: "Robert Fagles" }, contribution: "Translator" }] },
    { id: 3, title: "The Iliad", subtitle: "A new translation", users_count: 1200, language: { language: "English" } },
  ],
};

describe("hardcover", () => {
  it("sends the token with Bearer once, however it was pasted", () => {
    expect(hardcoverAuth("abc")).toBe("Bearer abc");
    expect(hardcoverAuth(" Bearer abc ")).toBe("Bearer abc");
  });

  it("ranks a book's editions by readers, Reflect's language first, with translators", () => {
    const b = bookOf(row)!;
    expect(b).toMatchObject({ id: 7, rating: 4.12, ratings: 900, readers: 5000 });
    expect(b.editions.map((e) => e.id)).toEqual([3, 2, 1]);
    expect(b.editions[0]!.title).toBe("The Iliad: A new translation");
    expect(b.editions[1]).toMatchObject({ isbn: "9780140275360", year: 1998, pages: 683, publisher: "Penguin", format: "Paperback", translators: ["Robert Fagles"], rating: 4.21 });
  });

  it("finds the book by most matched editions, or by a search hit with the same title and author", () => {
    expect(mostCommon([4, 9, 9, 4, 9])).toBe(9);
    expect(mostCommon([])).toBeNull();
    const results = { hits: [{ document: { id: 1, title: "The Iliad Companion", author_names: ["X"] } }, { document: { id: 2, title: "The Iliad", author_names: ["Someone Else"] } }, { document: { id: 3, title: "The Iliad", author_names: ["Homer"] } }] };
    expect(pickSearchHit(results, "The Iliad", "Homer")).toBe(3);
    expect(pickSearchHit(results, "The Odyssey", "Homer")).toBeNull();
    expect(translatorsOf([{ contribution: "Translator", author: { name: "A" } }, { author: { name: "B" } }])).toEqual(["A"]);
  });

  it("looks up by ISBN, falls back to search, and gives null when nothing matches", async () => {
    const calls: string[] = [];
    const q = async (query: string) => {
      calls.push(query.includes("editions(where: {_or") ? "isbn" : query.includes("search(") ? "search" : "book");
      if (query.includes("editions(where: {_or")) return { editions: [] };
      if (query.includes("search(")) return { search: { results: { hits: [{ document: { id: 7, title: "The Iliad", author_names: ["Homer"] } }] } } };
      return { books: [row] };
    };
    expect((await hardcoverBook({ title: "The Iliad", authors: ["Homer"], isbns: ["978-0140275360"] }, q))?.id).toBe(7);
    expect(calls).toEqual(["isbn", "search", "book"]);
    expect(await hardcoverBook({ title: "Nothing", authors: [], isbns: [] }, async () => ({ search: { results: { hits: [] } } }))).toBeNull();
    expect(await hardcoverBook({ title: "The Iliad", authors: [], isbns: ["9780140275360"] }, async () => { throw new Error("down"); })).toBeNull();
  });
});
