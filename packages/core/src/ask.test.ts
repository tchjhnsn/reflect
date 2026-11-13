import { describe, expect, it } from "vitest";
import { ASK_SYSTEM, cleanProposal, learnSummary, themaCode } from "./ask.ts";

describe("ask", () => {
  it("proposes a claim only with a stance it can keep, holders cleaned, never the person", () => {
    expect(cleanProposal({ text: "Marriage is a covenant", stance: "hold" })).toBeNull();
    expect(cleanProposal({ text: "Virtue is a mean", subject: "Ethics", stance: "belief", holders: [{ name: "User" }, { name: "Aristotle", citation: "NE II" }, { citation: "no name" }] })).toEqual({
      text: "Virtue is a mean",
      subject: "Ethics",
      stance: "belief",
      holders: [{ name: "Aristotle", citation: "NE II" }],
    });
  });

  it("takes only a Thema section or category Reflect knows", () => {
    expect([themaCode("t"), themaCode("jm"), themaCode("I"), themaCode("JZ"), themaCode(3)]).toEqual(["T", "JM", null, null, null]);
  });

  it("tells the model it never answers from memory or claims to save, and summarizes only titles and names", () => {
    expect(ASK_SYSTEM).toContain("never answer from memory");
    expect(ASK_SYSTEM).toContain("never say you recorded");
    expect(learnSummary({ books: [{ title: "Romans", status: "reading" }], subjects: ["The Bible"], claims: [{ text: "Marriage is a covenant", stance: "belief" }] })).toBe(
      "LEARN\nBooks: Romans (reading)\nSubjects: The Bible\nClaims: Marriage is a covenant (belief)",
    );
  });
});
