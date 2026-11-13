import { describe, expect, it } from "vitest";
import { ideasNaming, type Idea, type Stance, type Target } from "./ideas.ts";

describe("ideasNaming", () => {
  const idea = (id: string, from: Target[], maybe: Target[] = [], related: { t: Target; stance: Stance }[] = []): Idea => ({ id, title: id, stated: [], from: from.map((t) => ({ ...t, on: "2026-11-13" })), maybe: maybe.map((t) => ({ ...t, on: "2026-11-13" })), related: related.map((r) => ({ ...r.t, on: "2026-11-13", stance: r.stance, by: "me" })) });
  it("finds the ideas that came from a book, may have, or place it on the spectrum", () => {
    const all = [idea("a", [{ ref: "book:1" }]), idea("b", [], [{ text: "The Republic" }]), idea("c", [], [], [{ t: { ref: "book:1" }, stance: "challenges" }]), idea("d", [{ ref: "book:2" }])];
    expect(ideasNaming(all, "book:1", "The Republic").map((x) => `${x.idea.id}:${x.how}`)).toEqual(["a:from", "b:maybe", "c:challenges"]);
    expect(ideasNaming(all, null, "Meno")).toEqual([]);
  });
});
