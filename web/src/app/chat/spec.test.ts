import { describe, expect, it } from "vitest";
import { buildSpec } from "./spec";

describe("chat cards", () => {
  it("shows nothing when there is nothing to show", () => {
    expect(buildSpec([])).toBeNull();
  });

  it("builds library cards that carry their catalog entry for Add, and marks kept books", () => {
    const spec = buildSpec([{ kind: "library", label: '"wiring"', kept: new Set(["OL2W"]), books: [
      { work: "OL1W", title: "Wiring", authors: ["A. Writer"], subjects: [], contents: [], lcc: ["TK-1362"] },
      { work: "OL2W", title: "Kept", authors: [], subjects: [], contents: [] },
    ] }])!;
    expect(spec.root).toBe("root");
    expect(spec.elements.root!.children).toEqual(["e0", "e1"]);
    const a = spec.elements.e0!;
    expect(a.props).toMatchObject({ title: "Wiring", by: "A. Writer", shelf: "T Technology", onShelves: false, entry: expect.stringContaining('"work":"OL1W"') });
    expect(spec.elements.e1!.props).toMatchObject({ by: "unknown author", shelf: "not yet shelved", onShelves: true });
  });

  it("proposes a claim the person accepts, and says plainly when a list is empty", () => {
    const spec = buildSpec([
      { kind: "propose_claim", text: "Virtue is a mean", stance: "belief", holders: [{ name: "Aristotle", citation: "NE II.6" }] },
      { kind: "claims", claims: [] },
      { kind: "shelves", shelves: [] },
    ])!;
    expect(spec.elements.e0!.props).toMatchObject({ text: "Virtue is a mean", holders: "Aristotle (NE II.6)", subject: null });
    expect(JSON.parse(String((spec.elements.e0!.props as { claim: string }).claim))).toEqual({ text: "Virtue is a mean", holders: [{ name: "Aristotle", citation: "NE II.6" }], stance: "belief" });
    expect(spec.elements.e1).toMatchObject({ type: "Note", props: { text: "You keep no claims yet." } });
    expect(spec.elements.e2!.type).toBe("Note");
  });
});
