import { describe, expect, it } from "vitest";
import { chaptersOf, gutenbergBook, rdfEnglish } from "./gutenberg.ts";

const book = [
  "The Project Gutenberg eBook of Test",
  "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***",
  "",
  "Contents",
  "",
  "BOOK I.",
  "",
  "CHAPTER I.",
  "",
  "It is a truth universally",
  "acknowledged, that _a single man_ in possession.",
  "",
  "Second paragraph--with a dash.",
  "",
  "[Illustration: a picture]",
  "",
  "[Illustration: the dance",
  "Chapter II.]",
  "",
  "A new chapter begins.",
  "",
  "And goes on.",
  "",
  "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***",
  "licence text",
].join("\r\n");

describe("gutenberg", () => {
  it("cuts the header and licence, joins lines into paragraphs and splits the text into chapters", () => {
    expect(chaptersOf(book)).toEqual([
      { title: "Book I · Chapter I", paragraphs: ["It is a truth universally acknowledged, that a single man in possession.", "Second paragraph\u2014with a dash."] },
      { title: "Chapter II", paragraphs: ["A new chapter begins.", "And goes on."] },
    ]);
  });

  it("cuts a text with no headings into parts", () => {
    const parts = chaptersOf(Array.from({ length: 100 }, (_, i) => `Paragraph ${i}.`).join("\n\n"));
    expect(parts.map((p) => p.title)).toEqual(["Part 1", "Part 2", "Part 3"]);
    expect(parts[2]!.paragraphs).toHaveLength(20);
  });

  it("fetches the plain text by id and refuses anything else", async () => {
    const seen: string[] = [];
    const f = async (url: string) => { seen.push(url); return { ok: true, status: 200, json: async () => ({}), text: async () => book }; };
    expect((await gutenbergBook("1342", f))?.chapters).toHaveLength(2);
    expect(seen).toEqual(["https://www.gutenberg.org/cache/epub/1342/pg1342.txt"]);
    expect(await gutenbergBook("../x", f)).toBeNull();
  });
});

describe("gutenberg records", () => {
  it("reads the English title of a text in another language", () => {
    expect(rdfEnglish("<dcterms:title>Ιλιάδα</dcterms:title>\n<pgterms:name>Homer</pgterms:name><pgterms:name>Palles, Alexandros</pgterms:name>\n<dcterms:alternative>Iliad</dcterms:alternative>")).toEqual({ title: "Iliad", author: "Homer" });
    expect(rdfEnglish("<pgterms:name>Tolstoy, Leo</pgterms:name><dcterms:alternative>War and Peace</dcterms:alternative>")).toEqual({ title: "War and Peace", author: "Leo Tolstoy" });
    expect(rdfEnglish("<dcterms:alternative>Ἰλιάς</dcterms:alternative>")).toBeNull();
  });
});
