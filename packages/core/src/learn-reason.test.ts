import { describe, expect, it } from "vitest";
import { isEndKind, reasonBody, reasonParts } from "./learn-reason";

describe("why a person is learning something", () => {
  it("writes the two answers under their headings and reads them back", () => {
    const body = reasonBody("  I keep meeting Hegel in what I read.\r\n", "To read the Phenomenology on my own.");
    expect(body).toBe("## What is causing me to learn this\n\nI keep meeting Hegel in what I read.\n\n## For the sake of what\n\nTo read the Phenomenology on my own.\n");
    expect(reasonParts(body)).toEqual({ cause: "I keep meeting Hegel in what I read.", end: "To read the Phenomenology on my own." });
  });
  it("keeps text written around the headings in the journal", () => {
    expect(reasonParts("A note first.\n\n## What is causing me to learn this\n\nWonder.\n\n## For the sake of what\n\nItself.")).toEqual({ cause: "A note first.\n\nWonder.", end: "Itself." });
    expect(reasonParts("Just this.")).toEqual({ cause: "Just this.", end: "" });
  });
  it("knows the two kinds of end", () => {
    expect([isEndKind("own-sake"), isEndKind("use"), isEndKind("fun")]).toEqual([true, true, false]);
  });
});
