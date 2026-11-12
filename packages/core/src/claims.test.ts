import { describe, expect, it } from "vitest";
import { cleanClaim, currentStance, isStance, STANCES } from "./claims.ts";

describe("claims", () => {
  it("uses the three doxastic attitudes", () => {
    expect(STANCES.map((s) => s.id)).toEqual(["belief", "disbelief", "suspension"]);
    expect(isStance("hold")).toBe(false);
  });

  it("counts the latest stance, and none until the person says", () => {
    expect(currentStance([])).toBeNull();
    const s = currentStance([
      { stance: "belief", at: "2026-11-01T00:00:00Z" },
      { stance: "suspension", at: "2026-11-12T00:00:00Z" },
      { stance: "disbelief", at: "2026-11-05T00:00:00Z" },
    ]);
    expect(s!.stance).toBe("suspension");
  });

  it("cleans a claim and its cited holders", () => {
    const c = cleanClaim("  Marriage   is a covenant ", [{ name: " Scripture ", citation: "Malachi 2:14" }, { name: "  " }]);
    expect(c).toEqual({ text: "Marriage is a covenant", holders: [{ name: "Scripture", citation: "Malachi 2:14" }] });
    expect(() => cleanClaim(" ", [])).toThrow();
  });
});
