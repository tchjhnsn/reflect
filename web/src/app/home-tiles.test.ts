import { describe, expect, it } from "vitest";
import { GLYPHS, hrefOf, MINE, OURS, toneOf, type Facts } from "./home-tiles";

const none: Facts = { learn: { underway: false, stated: false }, journal: { stated: false, open: false } };

describe("home tiles", () => {
  it("has 15 tiles of mine (journal, ideas and legacy are a tool and lenses, not tiles) and 6 of ours, each with a glyph", () => {
    expect(MINE).toHaveLength(15);
    expect(MINE.map((t) => t.id)).not.toContain("journal");
    expect(OURS).toHaveLength(6);
    for (const t of [...MINE, ...OURS]) expect(GLYPHS[t.id], t.id).toBeTruthy();
  });

  it("opens every tile of mine, at the part it names, and none of ours yet", () => {
    const links = Object.fromEntries([...MINE, ...OURS].map((t) => [t.id, hrefOf(t)]));
    expect(links).toMatchObject({ learn: "/learn", food: "/life/home?k=food", income: "/life/work?k=income", standing: "/life/polis?k=standing", marriage: null });
    for (const t of MINE) expect(hrefOf(t), t.id).not.toBeNull();
  });

  it("takes tone from what the person has, unstated by default", () => {
    const learn = MINE.find((t) => t.id === "learn")!;
    expect(toneOf(learn, none)).toBe("unstated");
    expect(toneOf(learn, { ...none, learn: { underway: false, stated: true } })).toBe("thought");
    expect(toneOf(learn, { ...none, learn: { underway: true, stated: true } })).toBe("action");
    expect(toneOf(MINE.find((t) => t.id === "food")!, { learn: { underway: true, stated: true }, journal: { stated: true, open: true } })).toBe("unstated");
    expect(toneOf(MINE.find((t) => t.id === "food")!, { ...none, life: { food: "thought" } })).toBe("thought");
  });
});
