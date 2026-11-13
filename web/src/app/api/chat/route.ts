// The chat's endpoint, on the pattern of json-render's chat example (vercel-labs/json-render,
// examples/chat): AI SDK streaming, tools first, cards streamed as json-render spec parts. One difference
// on purpose: the model never writes a spec. Each tool fetches the person's records and Reflect builds the
// card (../../chat/spec.ts), so a card can only show what the app holds.
import { convertToModelMessages, createUIMessageStream, createUIMessageStreamResponse, stepCountIs, streamText, tool, type UIMessage } from "ai";
import { SPEC_DATA_PART_TYPE } from "@json-render/core";
import { z } from "zod";
import { ASK_SYSTEM, cleanProposal, learnSummary, themaCode } from "@reflect/core/ask";
import { STANCES } from "@reflect/core/claims";
import { browseQuery, searchBooks } from "@reflect/core/openlibrary";
import { themaCategory, themaQuery, themaSection } from "@reflect/core/thema";
import { acceptLocalDate, clean } from "@reflect/core/journal";
import { chatPart, isChatId } from "@reflect/core/chat-journal";
import { isoDate } from "@reflect/core/person";
import { currentUser } from "@/server/session";
import { keepChatMessage } from "@/server/journal-store";
import { chatModel } from "@/server/chat-model";
import { learnView } from "@/server/learn-store";
import { listClaims, type ClaimOut } from "@/server/claims-store";
import { shelvesOf } from "@/server/learn-shelves";
import { buildSpec, type Piece } from "../../chat/spec";

export const maxDuration = 60;
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const MAX_MESSAGES = 30;

export async function POST(req: Request) {
  const user = await currentUser();
  if (!user) return Response.json({ error: "sign in first" }, { status: 401 });
  const body = (await req.json().catch(() => null)) as { id?: unknown; localDate?: unknown; messages?: UIMessage[] } | null;
  const messages = Array.isArray(body?.messages) ? body.messages.slice(-MAX_MESSAGES) : [];
  if (!messages.length) return Response.json({ error: "messages are required" }, { status: 400 });
  const model = chatModel();
  if (!model) return Response.json({ error: "no model is set up" }, { status: 503 });

  const [view, claims] = await Promise.all([learnView(user.id), listClaims(user.id).catch((): ClaimOut[] | null => null)]);
  const summary = learnSummary({
    books: view.books.map((b) => ({ title: b.title, status: b.status })),
    subjects: [...new Set(view.books.flatMap((b) => b.subjects.map((s) => s.name)).concat(view.known.map((s) => s.name)))],
    claims: (claims ?? []).map((c) => ({ text: c.text, stance: c.stance?.stance ?? null })),
  });
  // The chat is kept in the journal as one entry (its id is the chat's id), one part per message.
  const day = typeof body?.localDate === "string" ? isoDate(body.localDate) : null;
  const journal = isChatId(body?.id) && day && acceptLocalDate(day, new Date()) ? { chat: body.id as string, day } : null;
  const keep = async (who: "you" | "reflect", text: string) => {
    const words = clean(text);
    if (journal && words) await keepChatMessage(user.id, journal.chat, journal.day, chatPart(who, words)).catch(() => undefined);
  };
  const last = messages.at(-1);
  if (last?.role === "user") await keep("you", last.parts.map((p) => (p.type === "text" ? p.text : "")).join(""));

  const kept = new Set(view.books.map((b) => b.catalog?.work).filter((w): w is string => Boolean(w)));

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      // Every card of this answer, redrawn as one spec each time a tool adds one.
      const pieces: Piece[] = [];
      const show = (p: Piece) => {
        pieces.push(p);
        const spec = buildSpec(pieces);
        if (spec) writer.write({ type: SPEC_DATA_PART_TYPE, id: "cards", data: { type: "flat", spec } });
      };
      const library = async (label: string, run: () => Promise<Awaited<ReturnType<typeof searchBooks>>>, href: string) => {
        const books = await run().catch(() => null);
        if (!books) {
          show({ kind: "note", text: "The open library did not answer." });
          return "The open library did not answer.";
        }
        show({ kind: "library", label, books, kept, href });
        return books.length ? `Shown: ${books.map((b) => `${b.title} by ${b.authors.join(", ") || "unknown"}${kept.has(b.work) ? " (already on their shelves)" : ""}`).join("; ")}` : "Nothing found.";
      };
      const result = streamText({
        model,
        system: `${ASK_SYSTEM}\n\n${summary}`,
        messages: await convertToModelMessages(messages),
        stopWhen: stepCountIs(4),
        onFinish: async ({ steps }) => keep("reflect", steps.map((st) => st.text).filter(Boolean).join("\n\n")),
        tools: {
          show_shelves: tool({
            description: "Show the person's shelves: their books and subjects by Thema section.",
            inputSchema: z.object({}),
            execute: async () => {
              const names = new Map(view.books.flatMap((b) => b.subjects).map((x) => [x.id, x.name]));
              const sh = shelvesOf(view).shelves.map((x) => ({ id: x.id, name: x.name, books: x.books.map((b) => b.title), subjects: x.subjects.map((y) => names.get(y.id) ?? y.name) }));
              show({ kind: "shelves", shelves: sh });
              return sh.length ? `Shown: ${sh.map((s) => `${s.id} ${s.name}: ${[...s.books, ...s.subjects].join(", ")}`).join("; ")}` : "No shelves yet.";
            },
          }),
          show_claims: tool({
            description: "Show the claims the person keeps, with their stance on each.",
            inputSchema: z.object({}),
            execute: async () => {
              if (!claims) {
                show({ kind: "note", text: "Claims are not set up here yet." });
                return "Claims are not set up.";
              }
              show({ kind: "claims", claims: claims.map((c) => ({ text: c.text, stance: c.stance?.stance ?? null, holders: c.holders })) });
              return claims.length ? `Shown: ${claims.map((c) => `${c.text} (${c.stance?.stance ?? "no stance"})`).join("; ")}` : "They keep no claims yet.";
            },
          }),
          search_library: tool({
            description: "Search the open library by title, author or topic; the person can add a result to their shelves.",
            inputSchema: z.object({ query: z.string().min(1).max(200) }),
            execute: async ({ query }) => library(`"${query}"`, () => searchBooks(query, openLibrary, 5), `/learn/browse?q=${encodeURIComponent(query)}`),
          }),
          browse_category: tool({
            description: "Show the best rated books in one Thema section (one letter) or category (two letters).",
            inputSchema: z.object({ code: z.string().min(1).max(2).describe("A Thema code, such as F (fiction), FM (fantasy), JM (psychology), JH (sociology), QR (religion)") }),
            execute: async (input) => {
              const c = themaCode(input.code);
              const q = c ? themaQuery(c) : null;
              if (!c || !q) return "That is not a Thema section or category Reflect knows.";
              return library((themaCategory(c) ?? themaSection(c))!.short, () => browseQuery(q, openLibrary, { limit: 5 }), `/learn/browse?cat=${c}`);
            },
          }),
          propose_claim: tool({
            description: "Propose a claim for the person to keep, only when they stated it and their stance in this conversation. They accept or ignore it.",
            inputSchema: z.object({
              text: z.string().min(1).max(500),
              subject: z.string().max(120).optional(),
              holders: z.array(z.object({ name: z.string(), citation: z.string().optional() })).max(10).default([]),
              stance: z.enum(STANCES.map((s) => s.id) as [string, ...string[]]),
            }),
            execute: async (input) => {
              const p = cleanProposal(input);
              if (!p) return "That claim could not be proposed.";
              show({ kind: "propose_claim", ...p });
              return "Proposed. It is not kept unless the person taps Accept.";
            },
          }),
        },
      });
      writer.merge(result.toUIMessageStream());
    },
    onError: () => "The model is not reachable right now. Your apps are all still there.",
  });
  return createUIMessageStreamResponse({ stream });
}
