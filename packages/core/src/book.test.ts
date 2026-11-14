import { describe, expect, it } from "vitest";
import { bookEntry, bookMarkdown, checkKind, compileBook, DOMAINS, type Belief } from "./book";
import { cleanOrigins } from "./origins";

const b = (id: string, domain: Belief["domain"], stances: [string, string][], extra: Partial<Belief> = {}): Belief => ({
  id, text: `claim ${id}`, holders: [], domain, source: null, history: stances.map(([stance, at]) => ({ stance: stance as "belief", at })), ...extra,
});

describe("the book of your life", () => {
  it("keeps every chapter in chart order, held beliefs first, revisions after", () => {
    const { chapters, unplaced } = compileBook([
      b("1", "line", [["belief", "2026-01-02"]]),
      b("2", "line", [["belief", "2025-01-01"], ["disbelief", "2026-03-01"]]),
      b("3", "line", [["belief", "2024-05-05"]]),
      b("4", null, [["belief", "2026-01-01"]]),
      b("5", "work", [["suspension", "2026-01-01"]]),
    ]);
    expect(chapters.map((c) => c.domain)).toEqual(DOMAINS.map((d) => d.id));
    const line = chapters.find((c) => c.domain === "line")!;
    expect(line.entries.map((e) => [e.id, e.state])).toEqual([["3", "held"], ["1", "held"], ["2", "once"]]);
    expect(chapters.find((c) => c.domain === "work")!.entries).toEqual([]);
    expect(unplaced.map((e) => e.id)).toEqual(["4"]);
  });

  it("a belief dropped and held again is held since it was first held", () => {
    expect(bookEntry(b("x", "work", [["belief", "2020-01-01"], ["suspension", "2021-01-01"], ["belief", "2022-01-01"]]))).toMatchObject({ state: "held", since: "2020-01-01" });
    expect(bookEntry(b("y", "work", [["disbelief", "2020-01-01"]]))).toBeNull();
  });

  it("writes an edition with sources, citations and revisions, skipping empty chapters", () => {
    const { chapters } = compileBook([
      b("1", "line", [["belief", "2026-01-02T10:00:00Z"]], { text: "Marriage is a covenant.", holders: [{ name: "Augustine", citation: "On the Good of Marriage" }] }),
      b("2", "work", [["belief", "2025-01-01"], ["disbelief", "2026-03-01"]], { text: "Work is only for money.", source: { workId: "w", title: "On work" } }),
    ]);
    const md = bookMarkdown(chapters, "2026-11-12T20:00:00Z");
    expect(md).toContain("*As of 2026-11-12.*");
    expect(md).toContain("## Work\n\n### Once held\n\n- ~~Work is only for money.~~ *(held 2025-01-01 to 2026-03-01)*");
    expect(md).toContain("- Marriage is a covenant. *(since 2026-01-02, with Augustine, On the Good of Marriage)*");
    expect(md).not.toContain("## Needs");
    expect(bookMarkdown(compileBook([]).chapters, "2026-11-12")).toContain("Nothing placed");
  });

  it("puts ideals before practical beliefs before unmarked ones, and writes what a practical belief serves", () => {
    const ideal = { id: "i", text: "Children are best raised by both parents." };
    const { chapters } = compileBook([
      b("u", "line", [["belief", "2020-01-01"]], { text: "Unmarked." }),
      b("p", "line", [["belief", "2021-01-01"]], { text: "We both work part time this year.", kind: "practical", serves: ideal, circumstances: "while the baby is under one", origins: ["reasoning", "compulsion"] }),
      b("i", "line", [["belief", "2025-01-01"]], { text: ideal.text, kind: "ideal", origins: ["nature"] }),
    ]);
    const line = chapters.find((c) => c.domain === "line")!;
    expect(line.entries.map((e) => e.id)).toEqual(["i", "p", "u"]);
    const md = bookMarkdown(chapters, "2026-11-12");
    expect(md).toContain("### Ideals\n\n- Children are best raised by both parents. *(since 2025-01-01, from nature)*");
    expect(md).toContain('### Practical beliefs\n\n- We both work part time this year. *(since 2021-01-01, serving "Children are best raised by both parents.", for while the baby is under one, from reasoning, compulsion)*');
    expect(md).toContain("### Not yet marked\n\n- Unmarked.");
  });

  it("a practical belief needs the ideal it serves and its circumstances; an ideal needs neither", () => {
    expect(checkKind("ideal", null, "ignored")).toEqual({ kind: "ideal", circumstances: null });
    expect(() => checkKind("practical", null, "now")).toThrow("Name the ideal it serves.");
    expect(() => checkKind("practical", "x", "  ")).toThrow("Say the circumstances it is for.");
    expect(checkKind("practical", "x", " while  renting ")).toEqual({ kind: "practical", circumstances: "while renting" });
    expect(() => checkKind("hold", null, null)).toThrow("Mark it as an ideal or a practical belief.");
  });

  it("origins are Aristotle's seven, once each, in his order", () => {
    expect(cleanOrigins(["appetite", "chance", "appetite", "whim"])).toEqual(["chance", "appetite"]);
    expect(cleanOrigins(["anger", "spirit", "habit"])).toEqual(["habit", "spirit"]);
  });
});
