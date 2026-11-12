import { describe, expect, it } from "vitest";
import { bookTone, subjectTone } from "./tones";

describe("shelf tones", () => {
  it("takes a book's tone from where the person is with it, and a subject's from their grasp", () => {
    expect([bookTone("read"), bookTone("reading"), bookTone("want")]).toEqual(["known", "learning", "registered"]);
    expect(subjectTone("gist", [])).toBe("known");
    expect(subjectTone("forgot", ["reading"])).toBe("learning");
    expect(subjectTone(null, ["want"])).toBe("registered");
  });
});
