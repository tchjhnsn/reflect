import { describe, expect, it } from "vitest";
import { bookWay, isMedium, isWay, MEDIA, placeOf } from "./ways.ts";

describe("ways of learning", () => {
  it("puts stories, poetry and drama under imitation and other books under teaching", () => {
    expect(bookWay("FM")).toBe("imitation");
    expect(bookWay("DC")).toBe("imitation");
    expect(bookWay("YF")).toBe("imitation");
    expect(bookWay("QD")).toBe("teaching");
    expect(bookWay(null)).toBe("teaching");
  });

  it("places an item by the person's way, else its medium's", () => {
    expect(placeOf({ medium: "lecture" })).toBe("teaching");
    expect(placeOf({ medium: "debate" })).toBe("dialectic");
    expect(placeOf({ medium: "film" })).toBe("imitation");
    expect(placeOf({ medium: "podcast", way: "teaching" })).toBe("teaching");
  });

  it("keeps games, music and art under rest and leisure until a lesson is said", () => {
    expect(placeOf({ medium: "game" })).toBe("leisure");
    expect(placeOf({ medium: "game", lesson: "  " })).toBe("leisure");
    expect(placeOf({ medium: "game", lesson: "How to plan three moves ahead" })).toBe("practice");
    expect(placeOf({ medium: "music", lesson: "What grief sounds like", way: "imitation" })).toBe("imitation");
    expect(placeOf({ medium: "art", way: "teaching" })).toBe("leisure");
  });

  it("knows its media and ways", () => {
    expect(MEDIA.every((m) => isWay(m.way))).toBe(true);
    expect(isMedium("game")).toBe(true);
    expect(isMedium("book")).toBe(false);
    expect(isWay("leisure")).toBe(false);
  });
});
