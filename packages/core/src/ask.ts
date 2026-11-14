// The chat behind the plus button (docs/design/REFLECT_UI_2026-11-12.md section 5). The model never answers
// from memory and never writes a record: it talks, and calls the apps' tools to show real records. Reflect
// builds every card from what the tool fetched, so a card can only show what the app holds. What would
// change a record is a proposal the person accepts with one tap. Facts about the person are never inferred.
// This module is pure: the instructions, the tools' inputs, and the summary the model may read.
import { isStance, STANCES, type Holder, type Stance } from "./claims.ts";
import { THEMA, themaCategory, themaSection } from "./thema.ts";

export const ASK_SYSTEM = `You are the chat inside Reflect, a person's private life app. Talk plainly, in two or three sentences of plain text: no markdown, no lists, no bold.

You never answer from memory about the person. What you know of them is only the LEARN summary below, from their records. To show real records, call a tool; the app draws a card from each call:
- show_shelves: their shelves (books and subjects by Thema section).
- show_ways: what they learn from by the ways of learning (teaching, dialectic, imitation, practice, and rest and leisure): books and items such as lectures, debates, films, music and games, with each item's medium.
- find_item(medium, query): look up a lecture, debate, podcast, film, series, play, music, art or game (a link, or a search of a free catalog) for them to add with one tap. Only what it returns exists; never name one from memory as if found.
- quote_transcript(words, item?): find words in the transcripts they pasted for their videos and recordings, with the moment of each place. Quote only from what it returns.
- show_claims: claims they keep, with their stance on each.
- search_library(query): books from the open library they can add to their shelves.
- browse_category(code): the best rated books in one Thema section (one letter) or category (two letters), such as ${THEMA.map((s) => `${s.code} ${s.short}`).join(", ")}; JM is psychology, JH sociology, FM fantasy.
- quote_book(words, book?): find words in the text of a book they are reading in Reflect, or the one they name, with the chapter of each place. Quote a book only from what it returns, word for word, and cite the author, book and chapter it gives.
- propose_claim(text, subject?, holders, stance): a claim to keep, only when the person stated it and their stance (${STANCES.map((s) => s.id).join(", ")}) in this conversation. "holders" are the thinkers, texts or traditions that hold the view, never the person themself.

Nothing is kept until the person taps Accept or Add, so never say you recorded, saved or kept anything. Never invent books, authors, citations or facts about the person; to name a book, search the library. Do not guess their stance or beliefs; ask. Suggest, never decide. Call at most three tools.`;

// The person's own stance is the stance line; a holder is someone else who holds the view.
const SELF = /^(the )?(user|person|me|you|i|myself|yourself)$/i;
const str = (v: unknown, max: number): string | null => (typeof v === "string" && v.trim() ? v.trim().slice(0, max) : null);

/** A proposed claim from the model, kept to what Reflect can keep, or null. The person is never a holder. */
export function cleanProposal(raw: unknown): { text: string; subject?: string; holders: Holder[]; stance: Stance } | null {
  const it = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const text = str(it.text, 500);
  const stance = typeof it.stance === "string" && isStance(it.stance) ? it.stance : null;
  if (!text || !stance) return null;
  const holders = (Array.isArray(it.holders) ? it.holders : [])
    .map((h) => (h && typeof h === "object" ? (h as Record<string, unknown>) : {}))
    .map((h) => ({ name: str(h.name, 200), citation: str(h.citation, 500) }))
    .filter((h): h is { name: string; citation: string | null } => Boolean(h.name) && !SELF.test(h.name!))
    .slice(0, 10)
    .map((h) => (h.citation ? { name: h.name, citation: h.citation } : { name: h.name }));
  const subject = str(it.subject, 120);
  return { text, ...(subject ? { subject } : {}), holders, stance };
}

/** A Thema section (one letter) or category (two letters) Reflect knows, or null. */
export function themaCode(raw: unknown): string | null {
  const c = typeof raw === "string" ? raw.trim().toUpperCase() : "";
  return (c.length === 1 && themaSection(c)) || (c.length === 2 && themaCategory(c)) ? c : null;
}

/** The LEARN summary the model may read: titles and names only, from the person's records. */
export function learnSummary(l: { books: { title: string; status: string }[]; subjects: string[]; claims: { text: string; stance: string | null }[]; items?: { title: string; medium: string; status: string }[] }): string {
  const cut = <T,>(xs: T[], n: number) => xs.slice(0, n);
  return [
    "LEARN",
    `Books: ${cut(l.books, 40).map((b) => `${b.title} (${b.status})`).join("; ") || "none"}`,
    `Subjects: ${cut(l.subjects, 60).join("; ") || "none"}`,
    `Claims: ${cut(l.claims, 40).map((c) => `${c.text} (${c.stance ?? "no stance"})`).join("; ") || "none"}`,
    ...(l.items ? [`Lectures, debates, films, music, games and more: ${cut(l.items, 40).map((i) => `${i.title} (${i.medium}, ${i.status})`).join("; ") || "none"}`] : []),
  ].join("\n");
}
