import { describe, expect, it } from "vitest";
import { ASK_SYSTEM, learnSummary, readPlan } from "./ask.ts";

describe("ask", () => {
  it("keeps only what Reflect can show, at most three", () => {
    const p = readPlan({
      say: "Here is what you have.",
      show: [
        { kind: "shelves" },
        { kind: "library_class", class: "t" },
        { kind: "library_class", class: "I" },
        { kind: "delete_everything" },
        { kind: "library", query: "electrical wiring" },
        { kind: "claims" },
      ],
    });
    expect(p.say).toBe("Here is what you have.");
    expect(p.show).toEqual([{ kind: "shelves" }, { kind: "library_class", class: "T" }, { kind: "library", query: "electrical wiring" }]);
  });

  it("proposes a claim only with a stance it can keep, holders cleaned", () => {
    expect(readPlan({ say: "x", show: [{ kind: "propose_claim", text: "Marriage is a covenant", stance: "hold" }] }).show).toEqual([]);
    const p = readPlan({ say: "x", show: [{ kind: "propose_claim", text: "Marriage is a covenant", subject: "Marriage", stance: "belief", holders: [{ name: "Scripture", citation: "Malachi 2:14" }, { citation: "no name" }] }] });
    expect(p.show).toEqual([{ kind: "propose_claim", text: "Marriage is a covenant", subject: "Marriage", stance: "belief", holders: [{ name: "Scripture", citation: "Malachi 2:14" }] }]);
  });

  it("answers plainly when the model says nothing usable", () => {
    expect(readPlan(null)).toEqual({ say: "I could not form an answer. Try saying it another way.", show: [] });
  });

  it("tells the model it never answers from memory, and summarizes only titles and names", () => {
    expect(ASK_SYSTEM).toContain("never answer from memory");
    expect(learnSummary({ books: [{ title: "Romans", status: "reading" }], subjects: ["The Bible"], claims: [{ text: "Marriage is a covenant", stance: "belief" }] })).toBe(
      "LEARN\nBooks: Romans (reading)\nSubjects: The Bible\nClaims: Marriage is a covenant (belief)",
    );
  });

  it("never lists the person as a holder of a claim; their stance is the stance", () => {
    const plan = readPlan({ say: "ok", show: [{ kind: "propose_claim", text: "Virtue is a mean", stance: "belief", holders: [{ name: "User" }, { name: "Aristotle", citation: "NE II" }] }] });
    expect(plan.show[0]).toMatchObject({ holders: [{ name: "Aristotle", citation: "NE II" }] });
  });
});
