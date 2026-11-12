import { describe, expect, it } from "vitest";
import { GLYPHS, hrefOf, MINE, OURS, toneOf, type Facts } from "./home-tiles";

const none: Facts = { learn: { underway: false, stated: false }, journal: { stated: false, open: false } };

describe("home tiles", () => {
  it("has 18 tiles of mine and 6 of ours, each with a glyph", () => {
    expect(MINE).toHaveLength(18);
    expect(OURS).toHaveLength(6);
    for (const t of [...MINE, ...OURS]) expect(GLYPHS[t.id], t.id).toBeTruthy();
  });

  it("links only apps built in the new design", () => {
    const links = Object.fromEntries([...MINE, ...OURS].map((t) => [t.id, hrefOf(t)]));
    expect(links).toMatchObject({ learn: "/learn", journal: "/journal", ideas: "/journal", legacy: null, food: null, marriage: null });
  });

  it("takes tone from what the person has, unstated by default", () => {
    const learn = MINE.find((t) => t.id === "learn")!;
    expect(toneOf(learn, none)).toBe("unstated");
    expect(toneOf(learn, { ...none, learn: { underway: false, stated: true } })).toBe("thought");
    expect(toneOf(learn, { ...none, learn: { underway: true, stated: true } })).toBe("action");
    expect(toneOf(MINE.find((t) => t.id === "food")!, { learn: { underway: true, stated: true }, journal: { stated: true, open: true } })).toBe("unstated");
    const journal = MINE.find((t) => t.id === "journal")!;
    expect(toneOf(journal, none)).toBe("unstated");
    expect(toneOf(journal, { ...none, journal: { stated: true, open: false } })).toBe("thought");
    expect(toneOf(journal, { ...none, journal: { stated: true, open: true } })).toBe("action");
  });
});
