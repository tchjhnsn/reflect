import { describe, expect, it } from "vitest";
import { editionsOfId, findFree, freeEditions, freeIndex } from "./free.ts";

const pg = freeIndex(["1342|pride and prejudice|austen jane", "3300|an inquiry into the nature and causes of the wealth of nations|smith adam", "2680|meditations|marcus aurelius emperor of rome", "999|the confessions of nat turner|turner nat"].join("\n"));

describe("free catalogs", () => {
  it("finds a book by every word of its title and a name of its author", () => {
    expect(findFree(pg, { title: "Pride and Prejudice", authors: ["Jane Austen"] })).toBe("1342");
    expect(findFree(pg, { title: "The Wealth of Nations", authors: ["Adam Smith"] })).toBe("3300");
    expect(findFree(pg, { title: "Meditations: A New Translation", authors: ["Marcus Aurelius"] })).toBe("2680");
  });

  it("matches nothing by another author, without an author, or with a word the catalog lacks", () => {
    expect(findFree(pg, { title: "Confessions", authors: ["Augustine of Hippo"] })).toBeNull();
    expect(findFree(pg, { title: "Pride and Prejudice" })).toBeNull();
    expect(findFree(pg, { title: "Pride and Prejudice and Zombies", authors: ["Jane Austen"] })).toBeNull();
  });

  it("finds each translation of a book", () => {
    const index = freeIndex("2199|the iliad|homer butler samuel|Samuel Butler\n6130|the iliad|homer pope alexander|Alexander Pope\n9|the iliad of homer for boys|church alfred\n10|the iliad|someone else");
    expect(freeEditions(index, { title: "The Iliad", authors: ["Homer"] })).toEqual([{ id: "2199", translators: "Samuel Butler" }, { id: "6130", translators: "Alexander Pope" }]);
    expect(freeEditions(index, { title: "The Iliad" })).toEqual([]);
    expect(editionsOfId(index, "6130").map((e) => e.id)).toEqual(["2199", "6130"]);
    expect(editionsOfId(index, "404")).toEqual([]);
  });
});
