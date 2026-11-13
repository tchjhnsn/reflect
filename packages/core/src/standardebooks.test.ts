import { describe, expect, it } from "vitest";
import { standardBook, standardChapters, standardNotes } from "./standardebooks.ts";

const html = `<section id="titlepage" epub:type="frontmatter titlepage"><h1>Title</h1><p>Front</p></section>
<section id="chapter-1" epub:type="bodymatter chapter z3998:fiction"><h2 epub:type="z3998:ordinal">I</h2><p>It is a truth <abbr>Mr.</abbr> Bennet&#8217;s &amp; <em>so</em> on.</p><p>Second.</p></section>
<section id="book-2" epub:type="bodymatter part"><hgroup><h2>Book II</h2><p epub:type="title">The Return</p></hgroup><section id="chapter-2" epub:type="chapter"><h3>II</h3><p>Inside.</p></section></section>`;

describe("standard ebooks", () => {
  it("reads chapters of the body matter as plain paragraphs, with their headings", () => {
    expect(standardChapters(html)).toEqual([
      { title: "Chapter I", paragraphs: ["It is a truth Mr. Bennet’s & so on.", "Second."] },
      { title: "Chapter II", paragraphs: ["Inside."] },
    ]);
  });

  it("asks only for a well-formed id", async () => {
    const seen: string[] = [];
    const f = async (u: string) => { seen.push(u); return { ok: true, status: 200, text: async () => html }; };
    expect(await standardBook("../x", f)).toBeNull();
    expect((await standardBook("jane-austen/pride-and-prejudice", f))?.chapters).toHaveLength(2);
    expect(seen).toEqual(["https://standardebooks.org/ebooks/jane-austen/pride-and-prejudice/text/single-page"]);
  });

  it("takes note marks out of the text, keeping where they sat, and reads the endnotes", () => {
    const withNotes = `<section id="chapter-1" epub:type="bodymatter chapter"><h2>I</h2><p>A word<a href="#note-1" id="noteref-1" epub:type="noteref">1</a> and <a href="#note-2" id="noteref-2" epub:type="noteref">2</a> more.</p></section>
<section id="endnotes" epub:type="backmatter endnotes"><ol><li id="note-1" epub:type="endnote"><p>First <i>note</i>. <a href="chapter-1.xhtml#noteref-1" epub:type="backlink">↩</a></p></li><li id="note-2"><p>Second.</p></li></ol></section>`;
    expect(standardChapters(withNotes)).toEqual([{ title: "Chapter I", paragraphs: ["A word and more."], notes: [{ p: 0, c: 6, id: "note-1" }, { p: 0, c: 11, id: "note-2" }] }]);
    expect(standardNotes(withNotes)).toEqual({ "note-1": "First note.", "note-2": "Second." });
    expect(standardNotes(html)).toEqual({});
  });
});
