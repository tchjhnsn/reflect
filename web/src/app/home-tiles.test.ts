import { describe, expect, it } from "vitest";
import { GLYPHS, hrefOf, MINE, OURS, toneOf, type Facts } from "./home-tiles";

const none: Facts = { learn: { underway: false, stated: false }, journal: { entries: 0 }, ideas: 0, legacy: false };

describe("home tiles", () => {
  it("has 18 tiles of mine and 6 of ours, each with a glyph", () => {
    expect(MINE).toHaveLength(18);
    expect(OURS).toHaveLength(6);
    for (const t of [...MINE, ...OURS]) expect(GLYPHS[t.id], t.id).toBeTruthy();
  });

  it("links only apps that exist", () => {
    const links = Object.fromEntries([...MINE, ...OURS].map((t) => [t.id, hrefOf(t)]));
    expect(links).toMatchObject({ learn: "/learn", journal: "/journal", ideas: "/ideas", legacy: "/legacy", food: null, marriage: null });
  });

  it("takes tone from what the person has, unstated by default", () => {
    const learn = MINE.find((t) => t.id === "learn")!;
    expect(toneOf(learn, none)).toBe("unstated");
    expect(toneOf(learn, { ...none, learn: { underway: false, stated: true } })).toBe("thought");
    expect(toneOf(learn, { ...none, learn: { underway: true, stated: true } })).toBe("action");
    expect(toneOf(MINE.find((t) => t.id === "journal")!, { ...none, journal: { entries: 2 } })).toBe("action");
    expect(toneOf(MINE.find((t) => t.id === "food")!, { ...none, journal: { entries: 2 } })).toBe("unstated");
  });
});
