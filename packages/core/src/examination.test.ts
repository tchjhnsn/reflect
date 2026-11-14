import { describe, expect, it } from "vitest";
import { EXAM, cleanText, joinRefusal } from "./examination";

describe("putting yourself under examination", () => {
  it("lets invited people join, never the candidate, and not once closed or full", () => {
    const e = { candidate: "ann", status: "open" as const, examiners: 1 };
    expect(joinRefusal(e, "ann", false)).toBe("own");
    expect(joinRefusal(e, "ben", false)).toBeNull();
    expect(joinRefusal({ ...e, status: "closed" }, "ben", false)).toBe("closed");
    expect(joinRefusal({ ...e, status: "closed" }, "ben", true)).toBeNull();
    expect(joinRefusal({ ...e, examiners: EXAM.examiners }, "cal", false)).toBe("full");
  });
  it("cleans what people write", () => {
    expect(cleanText("  a\r\nb  ", 10)).toBe("a\nb");
    expect(cleanText("   ", 10)).toBeNull();
    expect(cleanText(4, 10)).toBeNull();
    expect(cleanText("abcdef", 3)).toBe("abc");
  });
});
