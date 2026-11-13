import { describe, expect, it } from "vitest";
import { cleanWork, KINDS, TEMPLATES, templateById, workTitle } from "./works";

describe("works", () => {
  it("every template has a known kind and a unique id, blank first", () => {
    expect(TEMPLATES[0]?.id).toBe("blank");
    expect(new Set(TEMPLATES.map((t) => t.id)).size).toBe(TEMPLATES.length);
    for (const t of TEMPLATES) expect(KINDS[t.kind]).toBeDefined();
    expect(templateById("essay")?.kind).toBe("essay");
  });

  it("titles a work by its title, its first line, or its kind", () => {
    expect(workTitle({ title: " On friendship ", body: "", kind: "essay" })).toBe("On friendship");
    expect(workTitle({ title: "", body: "## Thesis\n\nFriends are made, not found.", kind: "essay" })).toBe("Thesis");
    expect(workTitle({ title: "", body: "", kind: "letter" })).toBe("Untitled letter");
  });

  it("cleans what is kept", () => {
    expect(cleanWork("  A   title ", "line  \r\nnext\n\n")).toEqual({ title: "A title", body: "line\nnext" });
    expect(cleanWork(1, null)).toEqual({ title: "", body: "" });
  });
});

describe("importing", () => {
  it("takes the title from a leading heading, else the file name", async () => {
    const { workFromFile } = await import("./works");
    expect(workFromFile("notes/on-friends.md", "\n# On friends\n\nThey are made.\n")).toEqual({ title: "On friends", body: "They are made." });
    expect(workFromFile("2019_diary.txt", "Rained all day.\r\n")).toEqual({ title: "2019 diary", body: "Rained all day." });
    expect(workFromFile("a.md", "---\ntags: x\n---\n# Not a title here")).toEqual({ title: "a", body: "---\ntags: x\n---\n# Not a title here" });
  });
});
