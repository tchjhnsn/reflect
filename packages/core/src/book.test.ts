import { describe, expect, it } from "vitest";
import { bookEntry, bookMarkdown, compileBook, DOMAINS, type Belief } from "./book";

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
    expect(md).toContain("## Work\n\n- ~~Work is only for money.~~ *(held 2025-01-01 to 2026-03-01)*");
    expect(md).toContain("- Marriage is a covenant. *(since 2026-01-02, with Augustine, On the Good of Marriage)*");
    expect(md).not.toContain("## Needs");
    expect(bookMarkdown(compileBook([]).chapters, "2026-11-12")).toContain("Nothing placed");
  });
});
