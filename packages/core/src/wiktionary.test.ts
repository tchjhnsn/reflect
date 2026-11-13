import { describe, expect, it } from "vitest";
import { definitionsOf, lookupWord } from "./wiktionary.ts";

const page = `==English==
===Etymology===
From Middle English.
===Adverb===
{{en-adv}}
# {{lb|en|conjunctive|interrogative|archaic}} [[why|Why]], for what reason, because of what.
#* {{RQ:Shakespeare|passage=wherefore art thou}}
# {{lb|en|archaic|or|formal}} [[therefore|Therefore]]; [[thus]].<ref>A ref</ref>
===Noun===
# The [[reason]] {{gloss|cause}} for something.
===Verb===
# {{past participle of|en|acknowledge}}
==French==
===Noun===
# Pas.`;

describe("wiktionary", () => {
  it("reads English senses with their labels and marks older meanings", () => {
    expect(definitionsOf(page)).toEqual([
      { pos: "Adverb", senses: [
        { text: "Why, for what reason, because of what.", labels: ["conjunctive", "interrogative", "archaic"], old: true },
        { text: "Therefore; thus.", labels: ["archaic", "formal"], old: true },
      ] },
      { pos: "Noun", senses: [{ text: "The reason (cause) for something.", labels: [], old: false }] },
      { pos: "Verb", senses: [{ text: "Past participle of acknowledge", labels: [], old: false, formOf: "acknowledge" }] },
    ]);
    expect(definitionsOf("==French==\n===Noun===\n# x")).toEqual([]);
  });

  it("takes one word from a selection", () => {
    expect(lookupWord("  “Wherefore,” ")).toBe("Wherefore");
    expect(lookupWord("don’t")).toBe("don't");
    expect(lookupWord("two words")).toBeNull();
    expect(lookupWord("1820")).toBeNull();
  });
});
