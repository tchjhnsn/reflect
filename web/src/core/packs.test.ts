import { describe, expect, it } from "vitest";
import { PACKS, drawPrompt, pool, promptById } from "./packs";

describe("prompt packs", () => {
  it("has unique prompt ids and no prescriptive wording", () => {
    const ids = PACKS.flatMap((p) => p.prompts.map((q) => q.id));
    expect(new Set(ids).size).toBe(ids.length);
    for (const q of PACKS.flatMap((p) => p.prompts)) {
      expect(q.text.endsWith("?")).toBe(true);
      expect(q.text).not.toMatch(/\b(should|must|will you)\b/i);
    }
  });

  it("draws only from chosen packs, and nothing when none is chosen", () => {
    expect(drawPrompt([], [], "2026-11-09", 0)).toBeNull();
    for (let t = 0; t < 30; t++) {
      expect(drawPrompt(["body"], [], "2026-11-09", t)?.id.startsWith("body.")).toBe(true);
    }
  });

  it("is deterministic and avoids used prompts until all are used", () => {
    expect(drawPrompt(["decisions"], [], "2026-11-09", 3)).toEqual(drawPrompt(["decisions"], [], "2026-11-09", 3));
    const all = pool(["decisions"]).map((p) => p.id);
    const used = all.slice(0, -1);
    expect(drawPrompt(["decisions"], used, "2026-11-09", 0)?.id).toBe(all.at(-1));
    expect(drawPrompt(["decisions"], all, "2026-11-09", 0)).not.toBeNull();
  });

  it("looks prompts up by id", () => {
    expect(promptById("body.1")?.text).toMatch(/body/);
    expect(promptById("nope")).toBeNull();
    expect(promptById(null)).toBeNull();
  });
});
