// Builds the json-render spec for one chat answer from what Reflect fetched out of its own apps. Pure, so
// every card can be checked: the model names what to show; only this file decides how it looks.
import type { Spec, UIElement } from "@json-render/core";
import type { Holder } from "@reflect/core/claims";
import type { CatalogEntry } from "@reflect/core/openlibrary";
import { categoryOf, themaCategory } from "@reflect/core/thema";

export type Piece =
  | { kind: "shelves"; shelves: { id: string; name: string; books: string[]; subjects: string[] }[] }
  | { kind: "claims"; claims: { text: string; stance: string | null; holders: Holder[] }[] }
  | { kind: "library"; label: string; books: CatalogEntry[]; kept: ReadonlySet<string> }
  | { kind: "propose_claim"; text: string; subject?: string; holders: Holder[]; stance: string }
  | { kind: "note"; text: string };

const holderText = (hs: readonly Holder[]) => hs.map((h) => (h.citation ? `${h.name} (${h.citation})` : h.name)).join("; ");
const shelfName = (e: Pick<CatalogEntry, "lcc" | "subjects">) => {
  const c = categoryOf(e);
  return c ? themaCategory(c)?.short ?? c : "not yet placed";
};

/** One spec for the cards of one answer, or null when there is nothing to show. */
export function buildSpec(pieces: readonly Piece[]): Spec | null {
  const elements: Record<string, UIElement> = {};
  const root: string[] = [];
  let n = 0;
  const add = (el: UIElement) => {
    const key = `e${n++}`;
    elements[key] = el;
    root.push(key);
    return key;
  };
  const note = (text: string) => add({ type: "Note", props: { text } });
  for (const p of pieces) {
    if (p.kind === "note") note(p.text);
    else if (p.kind === "shelves") {
      if (!p.shelves.length) note("No shelves yet. Books go on a shelf once they have a call number.");
      for (const s of p.shelves) add({ type: "Shelf", props: { id: s.id, name: s.name, books: s.books, subjects: s.subjects } });
    } else if (p.kind === "claims") {
      if (!p.claims.length) note("You keep no claims yet.");
      for (const c of p.claims) add({ type: "Claim", props: { text: c.text, stance: c.stance ?? "no stance", holders: holderText(c.holders) } });
    } else if (p.kind === "library") {
      if (!p.books.length) note(`The open library has nothing for ${p.label}.`);
      for (const b of p.books) {
        add({
          type: "LibraryBook",
          props: { title: b.title, by: b.authors.join(", ") || "unknown author", shelf: shelfName(b), onShelves: p.kept.has(b.work), entry: JSON.stringify(b) },
        });
      }
    } else {
      const claim = { text: p.text, holders: p.holders, stance: p.stance, ...(p.subject ? { subject: p.subject } : {}) };
      add({
        type: "ProposeClaim",
        props: { text: p.text, stance: p.stance, holders: holderText(p.holders), subject: p.subject ?? null, claim: JSON.stringify(claim) },
      });
    }
  }
  if (!root.length) return null;
  elements.root = { type: "Stack", props: {}, children: root };
  return { root: "root", elements };
}
