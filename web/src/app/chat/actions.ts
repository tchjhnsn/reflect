"use server";

import { revalidatePath } from "next/cache";
import { getVercelOidcToken } from "@vercel/oidc";
import type { Spec } from "@json-render/core";
import { ASK_SYSTEM, learnSummary, readPlan } from "@reflect/core/ask";
import { chatJsonMessages, type ChatMessage } from "@reflect/core/chat";
import { browseClass, cleanEntry, searchBooks } from "@reflect/core/openlibrary";
import { LCC_CLASSES } from "@reflect/core/shelves";
import { requireUser } from "@/server/session";
import { addBook, learnView } from "@/server/learn-store";
import { keepClaim, listClaims, type ClaimOut } from "@/server/claims-store";
import { shelvesOf } from "@/server/learn-shelves";
import { buildSpec, type Piece } from "./spec";

export type Turn = { role: "user" | "assistant"; content: string };
export type Answer = { say: string; spec: Spec | null };

const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const gateway = async () => (process.env.VERCEL ? ((await getVercelOidcToken().catch(() => undefined)) ?? null) : null);
const MAX_TURNS = 20;

/**
 * One answer in the chat. The model reads only the person's Learn summary and the conversation, and names
 * what to show; Reflect fetches each piece from its own apps and builds the cards. Nothing is written here.
 */
export async function askAction(history: Turn[]): Promise<Answer> {
  const user = await requireUser();
  const turns = (Array.isArray(history) ? history : [])
    .filter((t) => (t?.role === "user" || t?.role === "assistant") && typeof t.content === "string" && t.content.trim())
    .slice(-MAX_TURNS)
    .map((t): ChatMessage => ({ role: t.role, content: t.content.slice(0, 4000) }));
  if (!turns.length || turns[turns.length - 1]!.role !== "user") return { say: "Say something first.", spec: null };
  const [view, claims] = await Promise.all([learnView(user.id), listClaims(user.id).catch((): ClaimOut[] | null => null)]);
  const summary = learnSummary({
    books: view.books.map((b) => ({ title: b.title, status: b.status })),
    subjects: [...new Set(view.books.flatMap((b) => b.subjects.map((s) => s.name)).concat(view.known.map((s) => s.name)))],
    claims: (claims ?? []).map((c) => ({ text: c.text, stance: c.stance?.stance ?? null })),
  });
  const raw = await chatJsonMessages([{ role: "system", content: `${ASK_SYSTEM}\n\n${summary}` }, ...turns], await gateway()).catch(() => null);
  if (raw === null) return { say: "The model is not reachable right now, so I can't answer. Your apps are all still there.", spec: null };
  const plan = readPlan(raw);
  const kept = new Set(view.books.map((b) => b.catalog?.work).filter((w): w is string => Boolean(w)));
  const pieces = await Promise.all(
    plan.show.map(async (s): Promise<Piece> => {
      if (s.kind === "shelves") {
        const names = new Map(view.books.flatMap((b) => b.subjects).map((x) => [x.id, x.name]));
        const sh = shelvesOf(view);
        return { kind: "shelves", shelves: sh.shelves.map((x) => ({ id: x.id, name: x.name, books: x.books.map((b) => b.title), subjects: x.subjects.map((y) => names.get(y.id) ?? y.name) })) };
      }
      if (s.kind === "claims") {
        if (!claims) return { kind: "note", text: "Claims are not set up here yet." };
        return { kind: "claims", claims: claims.map((c) => ({ text: c.text, stance: c.stance?.stance ?? null, holders: c.holders })) };
      }
      if (s.kind === "library") {
        const books = await searchBooks(s.query, openLibrary, 5).catch(() => null);
        return books ? { kind: "library", label: `"${s.query}"`, books, kept } : { kind: "note", text: "The open library did not answer." };
      }
      if (s.kind === "library_class") {
        const books = await browseClass(s.class, openLibrary, { limit: 5 }).catch(() => null);
        const name = LCC_CLASSES.find((c) => c.id === s.class)?.name ?? s.class;
        return books ? { kind: "library", label: name, books, kept } : { kind: "note", text: "The open library did not answer." };
      }
      return { kind: "propose_claim", text: s.text, holders: s.holders, stance: s.stance, ...(s.subject ? { subject: s.subject } : {}) };
    }),
  );
  return { say: plan.say, spec: buildSpec(pieces) };
}

/** Add to my shelves, from a library card in the chat. The person tapped it; the entry is cleaned again. */
export async function addBookFromChat(entryJson: string): Promise<void> {
  const user = await requireUser();
  let raw: unknown;
  try {
    raw = JSON.parse(String(entryJson));
  } catch {
    throw new Error("that library entry could not be read");
  }
  const clean = cleanEntry(raw);
  if (!clean) throw new Error("that library entry has no title");
  await addBook(user.id, { title: clean.title, authors: clean.authors, ...(clean.isbn ? { isbn: clean.isbn } : {}), status: "want", subjects: [], catalog: clean });
  revalidatePath("/learn");
}

/** Accept, from a proposed claim in the chat. keepClaim checks the stance and cleans the text and holders. */
export async function keepClaimFromChat(claimJson: string): Promise<void> {
  const user = await requireUser();
  let c: { text?: unknown; holders?: unknown; stance?: unknown; subject?: unknown };
  try {
    c = JSON.parse(String(claimJson)) as typeof c;
  } catch {
    throw new Error("that claim could not be read");
  }
  const holders = (Array.isArray(c.holders) ? c.holders : [])
    .map((h) => (h && typeof h === "object" ? (h as Record<string, unknown>) : {}))
    .filter((h) => typeof h.name === "string")
    .map((h) => ({ name: String(h.name), ...(typeof h.citation === "string" ? { citation: h.citation } : {}) }));
  await keepClaim(user.id, { text: String(c.text ?? ""), holders, stance: String(c.stance ?? ""), ...(typeof c.subject === "string" ? { subject: c.subject } : {}) });
  revalidatePath("/learn");
}
