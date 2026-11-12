import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createStateStore } from "@json-render/core";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { buildSpec } from "./spec";

const added: string[] = [];
vi.mock("./actions", () => ({
  addBookFromChat: async (e: string) => void added.push(e),
  keepClaimFromChat: async () => {
    throw new Error("down");
  },
}));
const { registry, handlers } = await import("./registry");

describe("chat cards render", () => {
  it("renders library and claim cards from a spec, with the buttons until done and the note after", () => {
    const spec = buildSpec([
      { kind: "library", label: "x", kept: new Set(), books: [{ work: "OL1W", title: "Wiring", authors: ["A. Writer"], subjects: [], contents: [], lcc: ["TK-1"] }] },
      { kind: "propose_claim", text: "Virtue is a mean", stance: "belief", holders: [{ name: "Aristotle" }] },
    ])!;
    const html = (state: Record<string, unknown>) =>
      renderToStaticMarkup(
        <JSONUIProvider registry={registry} store={createStateStore(state)}>
          <Renderer spec={spec} registry={registry} />
        </JSONUIProvider>,
      );
    const before = html(spec.state ?? {});
    expect(before).toContain("Wiring");
    expect(before).toContain("T Technology");
    expect(before).toContain("Add to my shelves");
    expect(before).toContain("Accept");
    const after = html({ done: { e0: true, e1: true }, failed: {} });
    expect(after).toContain("on your shelves");
    expect(after).toContain("Kept in Learn.");
    expect(after).not.toContain("Accept<");
  });

  it("a tap runs the server action and marks its card done, or failed when the action throws", async () => {
    let state: Record<string, unknown> = { done: {}, failed: {} };
    const acts = handlers(() => (f) => (state = f(state)), () => state);
    await acts.add_book!({ key: "e0", entry: '{"work":"OL1W","title":"Wiring"}' });
    await acts.keep_claim!({ key: "e1", claim: "{}" });
    expect(added).toEqual(['{"work":"OL1W","title":"Wiring"}']);
    expect(state).toEqual({ done: { e0: true }, failed: { e1: true } });
  });
});
