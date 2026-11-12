// The chat behind the plus button (docs/design/REFLECT_UI_2026-11-12.md section 5). The model never answers
// from memory and never writes a record: it says a few words and names which pieces of the apps to show.
// Reflect fetches each piece from the app itself, so every card carries the person's real records. What
// would change a record is a proposal the person accepts with one tap. Facts about the person are never
// inferred. This module is pure: it holds the instructions, reads the model's plan and refuses anything
// outside it.
import { isStance, STANCES, type Holder, type Stance } from "./claims.ts";
import { LCC_CLASSES } from "./shelves.ts";

export type Show =
  | { kind: "shelves" }
  | { kind: "claims" }
  | { kind: "library"; query: string }
  | { kind: "library_class"; class: string }
  | { kind: "propose_claim"; text: string; subject?: string; holders: Holder[]; stance: Stance };
export type Plan = { say: string; show: Show[] };

const MAX_SHOW = 3;
// The person's own stance is the stance line; a holder is someone else who holds the view.
const SELF = /^(the )?(user|person|me|you|i|myself|yourself)$/i;

export const ASK_SYSTEM = `You are the chat inside Reflect, a person's private life app. Reply with JSON only:
{"say": "<two or three plain sentences>", "show": [<up to ${MAX_SHOW} items>]}

You never answer from memory about the person. What you know of them is only the LEARN summary below, from their records. To show real records, ask the app with one of these items:
- {"kind":"shelves"}: their shelves (books and subjects by Library of Congress class).
- {"kind":"claims"}: claims they keep, with their stance on each.
- {"kind":"library","query":"<title, author or topic>"}: books from the open library they can add to their shelves.
- {"kind":"library_class","class":"<one letter: ${LCC_CLASSES.map((c) => c.id).join(" ")}>"}: the best rated books in one class.
- {"kind":"propose_claim","text":"<the claim>","subject":"<optional>","holders":[{"name":"<who holds it>","citation":"<source>"}],"stance":"<${STANCES.map((s) => s.id).join("|")}>"}: a claim to keep, only when the person stated it and their stance in this conversation. Nothing is kept until they tap Accept, so never say you recorded, saved or kept it; say you can keep it if they accept. "holders" are the thinkers, texts or traditions that hold the view, never the person themself.

Rules: never invent books, authors, citations or facts about the person. Do not guess their stance or beliefs; ask. Name who holds a view when you mention one. Suggest, never decide. If nothing needs showing, "show" is [].`;

const str = (v: unknown, max: number): string | null => (typeof v === "string" && v.trim() ? v.trim().slice(0, max) : null);

/** The model's plan, kept to what Reflect can do. Anything else is dropped, never guessed at. */
export function readPlan(raw: unknown): Plan {
  const o = (raw && typeof raw === "object" ? raw : {}) as { say?: unknown; show?: unknown };
  const say = str(o.say, 1200) ?? "I could not form an answer. Try saying it another way.";
  const show: Show[] = [];
  for (const item of Array.isArray(o.show) ? o.show : []) {
    if (show.length >= MAX_SHOW) break;
    const it = (item && typeof item === "object" ? item : {}) as Record<string, unknown>;
    if (it.kind === "shelves" || it.kind === "claims") show.push({ kind: it.kind });
    else if (it.kind === "library") {
      const q = str(it.query, 200);
      if (q) show.push({ kind: "library", query: q });
    } else if (it.kind === "library_class") {
      const c = typeof it.class === "string" ? it.class.trim().toUpperCase() : "";
      if (LCC_CLASSES.some((x) => x.id === c)) show.push({ kind: "library_class", class: c });
    } else if (it.kind === "propose_claim") {
      const text = str(it.text, 500);
      const stance = typeof it.stance === "string" && isStance(it.stance) ? it.stance : null;
      if (!text || !stance) continue;
      const holders = (Array.isArray(it.holders) ? it.holders : [])
        .map((h) => (h && typeof h === "object" ? (h as Record<string, unknown>) : {}))
        .map((h) => ({ name: str(h.name, 200), citation: str(h.citation, 500) }))
        .filter((h): h is { name: string; citation: string | null } => Boolean(h.name) && !SELF.test(h.name!))
        .slice(0, 10)
        .map((h) => (h.citation ? { name: h.name, citation: h.citation } : { name: h.name }));
      const subject = str(it.subject, 120);
      show.push({ kind: "propose_claim", text, ...(subject ? { subject } : {}), holders, stance });
    }
  }
  return { say, show };
}

/** The LEARN summary the model may read: titles and names only, from the person's records. */
export function learnSummary(l: { books: { title: string; status: string }[]; subjects: string[]; claims: { text: string; stance: string | null }[] }): string {
  const cut = <T,>(xs: T[], n: number) => xs.slice(0, n);
  return [
    "LEARN",
    `Books: ${cut(l.books, 40).map((b) => `${b.title} (${b.status})`).join("; ") || "none"}`,
    `Subjects: ${cut(l.subjects, 60).join("; ") || "none"}`,
    `Claims: ${cut(l.claims, 40).map((c) => `${c.text} (${c.stance ?? "no stance"})`).join("; ") || "none"}`,
  ].join("\n");
}
