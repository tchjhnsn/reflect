import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { JSONUIProvider, Renderer } from "@json-render/react";
import { buildSpec } from "./spec";

vi.mock("./actions", () => ({ addBookFromChat: async () => {}, keepClaimFromChat: async () => {} }));
const { registry } = await import("./registry");

describe("chat cards render", () => {
  it("renders library and claim cards from a spec, with Add only for books not yet on the shelves", () => {
    const spec = buildSpec([
      { kind: "library", label: "x", kept: new Set(["OL2W"]), books: [
        { work: "OL1W", title: "Wiring", authors: ["A. Writer"], subjects: [], contents: [], lcc: ["TK-1"] },
        { work: "OL2W", title: "Kept", authors: [], subjects: [], contents: [] },
      ] },
      { kind: "propose_claim", text: "Virtue is a mean", stance: "belief", holders: [{ name: "Aristotle" }] },
    ])!;
    const html = renderToStaticMarkup(
      <JSONUIProvider registry={registry}>
        <Renderer spec={spec} registry={registry} />
      </JSONUIProvider>,
    );
    expect(html).toContain("Wiring");
    expect(html).toContain("T Technology");
    expect(html.match(/ADD TO SHELVES/g)).toHaveLength(1);
    expect(html).toContain("on your shelves");
    expect(html).toContain("ACCEPT");
    expect(html).toContain("held by Aristotle");
  });
});
