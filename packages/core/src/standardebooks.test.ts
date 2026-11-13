import { describe, expect, it } from "vitest";
import { standardBook, standardChapters } from "./standardebooks.ts";

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
});
