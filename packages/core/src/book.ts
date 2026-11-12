// The book of your life (Torian, 2026-11-12 18:39Z; built as proposed at 20:12Z). Beliefs the person adopts,
// stated in their own words or adopted with a citation, are compiled into the journal, one chapter per domain
// of the Ventures of a Lifetime chart, in chart order. The person places each belief in a domain; Reflect never
// guesses. A belief that was held and is no longer shows as a revision, so the book reads as a life that grew.
// The book is a live view; an edition is a dated snapshot of it, kept as a work.
import type { Holder, StanceLine } from "./claims";

/** The chapters, in the order of the Ventures of a Lifetime chart (the vault's folders, lifetime.ts). */
export const DOMAINS = [
  { id: "needs", title: "Needs", about: "Food, shelter, health, income: providing for yourself." },
  { id: "learn", title: "Learning", about: "What you know and how you came to know it." },
  { id: "work", title: "Work", about: "Work and the ventures you build." },
  { id: "belonging", title: "Belonging", about: "Friends, groups, and the standing you hold." },
  { id: "line", title: "Line", about: "Seeking a mate, marriage, children, caring for kin." },
  { id: "household", title: "Household", about: "The home and the people in it." },
  { id: "community", title: "Community", about: "Neighbors, congregation, the village." },
  { id: "polis", title: "Polis", about: "Civic life and the city." },
  { id: "legacy", title: "Legacy", about: "What you want to survive you." },
] as const;
export type Domain = (typeof DOMAINS)[number]["id"];
export const isDomain = (d: unknown): d is Domain => DOMAINS.some((x) => x.id === d);

/** A claim as the book reads it: its stance history, its domain if placed, and where it came from. */
export type Belief = {
  id: string;
  text: string;
  holders: Holder[];
  history: StanceLine[];
  domain: Domain | null;
  /** The passage in one of the person's works it was marked from, if any. */
  source: { workId: string; title: string } | null;
};

export type BookEntry = {
  id: string;
  text: string;
  holders: Holder[];
  source: Belief["source"];
  /** Held now, or once held and no longer. */
  state: "held" | "once";
  /** When it was first held, and when it stopped being held (for `once`). */
  since: string;
  until: string | null;
};
export type Chapter = { domain: Domain; title: string; about: string; entries: BookEntry[] };

/** One belief's place in the book, or null if it was never held (a disbelief or suspension only). */
export function bookEntry(b: Belief): BookEntry | null {
  const lines = [...b.history].sort((x, y) => x.at.localeCompare(y.at));
  const firstHeld = lines.find((l) => l.stance === "belief");
  if (!firstHeld) return null;
  const last = lines.at(-1)!;
  if (last.stance === "belief") return { id: b.id, text: b.text, holders: b.holders, source: b.source, state: "held", since: firstHeld.at, until: null };
  const lastHeld = [...lines].reverse().find((l) => l.stance === "belief")!;
  const dropped = lines.find((l) => l.at > lastHeld.at)!;
  return { id: b.id, text: b.text, holders: b.holders, source: b.source, state: "once", since: firstHeld.at, until: dropped.at };
}

/**
 * The book: a chapter per domain in chart order, held beliefs first then revisions, each oldest first.
 * Chapters with nothing in them are kept so the person sees the whole of a life. Beliefs held but not yet
 * placed in a domain are returned apart; they are not in the book until the person places them.
 */
export function compileBook(beliefs: readonly Belief[]): { chapters: Chapter[]; unplaced: BookEntry[] } {
  const entries = beliefs.map((b) => ({ b, e: bookEntry(b) })).filter((x): x is { b: Belief; e: BookEntry } => x.e !== null);
  const order = (a: BookEntry, z: BookEntry) => (a.state === z.state ? a.since.localeCompare(z.since) : a.state === "held" ? -1 : 1);
  return {
    chapters: DOMAINS.map((d) => ({ domain: d.id, title: d.title, about: d.about, entries: entries.filter((x) => x.b.domain === d.id).map((x) => x.e).sort(order) })),
    unplaced: entries.filter((x) => !x.b.domain).map((x) => x.e).sort(order),
  };
}

const day = (iso: string) => iso.slice(0, 10);
const cite = (h: Holder) => (h.citation ? `${h.name}, ${h.citation}` : h.name);

/** An edition of the book as Markdown, dated: what is kept as a work and could later be published. */
export function bookMarkdown(chapters: readonly Chapter[], asOf: string): string {
  const out: string[] = [`*As of ${day(asOf)}.*`];
  for (const c of chapters) {
    if (!c.entries.length) continue;
    out.push(`## ${c.title}`);
    for (const e of c.entries) {
      const from = [e.holders.length ? `with ${e.holders.map(cite).join("; ")}` : "", e.source ? `from "${e.source.title}"` : ""].filter(Boolean).join(", ");
      out.push(e.state === "held" ? `- ${e.text} *(since ${day(e.since)}${from ? `, ${from}` : ""})*` : `- ~~${e.text}~~ *(held ${day(e.since)} to ${day(e.until!)})*`);
    }
  }
  if (out.length === 1) out.push("Nothing placed in the book yet.");
  return out.join("\n\n");
}
