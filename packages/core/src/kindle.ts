// Kindle highlights from the "My Clippings.txt" file a Kindle keeps (in its documents folder). Each clipping is a
// book line "Title (Author)", a line saying what it is and where ("- Your Highlight on page 21 | Location 300-302 |
// Added on ..."), a blank line, its text, and a line of ten "=". Notes are joined to the highlight they were typed
// on (the same end location); bookmarks are dropped. Pure parsing: no I/O.

/** One highlight: the book it is from, where it is (page and location as the Kindle gives them), its words, a note typed on it. */
export type Clipping = { title: string; author: string; place: string; text: string; note?: string };

/** The book line: "Title (Author)", the author in the last parentheses, which may hold parentheses of their own. */
function bookOf(line: string): { title: string; author: string } {
  const t = line.replace(/^\uFEFF/, "").trim();
  if (t.endsWith(")")) {
    let depth = 0;
    for (let i = t.length - 1; i >= 0; i--) {
      if (t[i] === ")") depth++;
      else if (t[i] === "(" && --depth === 0) {
        const title = t.slice(0, i).trim();
        if (title) return { title, author: t.slice(i + 1, -1).trim() };
        break;
      }
    }
  }
  return { title: t, author: "" };
}

/** Where the clipping is, from its second line: "page 21, location 300-302", whichever the Kindle gives. */
function placeOf(line: string): { kind: "highlight" | "note" | "bookmark" | "other"; place: string; end: string } {
  const kind = /highlight/i.test(line) ? "highlight" : /\bnote\b/i.test(line) ? "note" : /bookmark/i.test(line) ? "bookmark" : "other";
  const page = /page\s+([\w-]+)/i.exec(line)?.[1];
  const loc = /location\s+([\d-]+)/i.exec(line)?.[1];
  const place = [page ? `page ${page}` : "", loc ? `location ${loc}` : ""].filter(Boolean).join(", ");
  return { kind, place, end: loc?.split("-").at(-1) ?? page ?? "" };
}

/** Every highlight in a My Clippings.txt file, in the file's order, each with any note typed on it. */
export function parseClippings(text: string): Clipping[] {
  const out: (Clipping & { end: string })[] = [];
  for (const block of text.replace(/\r\n?/g, "\n").split(/^==========\s*$/m)) {
    const lines = block.split("\n").map((l) => l.trim());
    while (lines.length && !lines[0]) lines.shift();
    if (lines.length < 2) continue;
    const book = bookOf(lines[0]!);
    const where = placeOf(lines[1]!);
    const body = lines.slice(2).join(" ").replace(/\s+/g, " ").trim();
    if (!body || !book.title) continue;
    if (where.kind === "highlight") {
      out.push({ ...book, place: where.place, text: body, end: where.end });
    } else if (where.kind === "note") {
      const on = [...out].reverse().find((c) => c.title === book.title && c.end === where.end);
      if (on) on.note = on.note ? `${on.note} ${body}` : body;
    }
  }
  // The Kindle writes a highlight again when it is changed; the last one stands.
  const seen = new Map<string, Clipping & { end: string }>();
  for (const c of out) seen.set(`${c.title}|${c.place}`, c);
  return [...seen.values()].map(({ end: _end, ...c }) => c);
}
