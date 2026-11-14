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
import { quoteFromShelves } from "@/server/passages";
import { buildSpec, type Piece } from "../../chat/spec";
import { getItem, listItems, type LearnItem } from "@/server/ways-store";
import { LEISURE, MEDIA, WAYS, bookWay, mediumOf, placeOf } from "@reflect/core/ways";
import { categoryOf } from "@reflect/core/thema";
import { lookLink, searchMedium, type Fetch } from "@reflect/core/lookup";
import { clock, readTranscript, searchTranscript } from "@reflect/core/media";

export const maxDuration = 60;
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const MAX_MESSAGES = 30;
const catalogFetch: Fetch = (url) => fetch(url, { headers: { "user-agent": "reflect-app (learn items)" }, next: { revalidate: 86400 }, signal: AbortSignal.timeout(8000) });

export async function POST(req: Request) {
  const user = await currentUser();
  if (!user) return Response.json({ error: "sign in first" }, { status: 401 });
  const body = (await req.json().catch(() => null)) as { id?: unknown; localDate?: unknown; messages?: UIMessage[] } | null;
  const messages = Array.isArray(body?.messages) ? body.messages.slice(-MAX_MESSAGES) : [];
  if (!messages.length) return Response.json({ error: "messages are required" }, { status: 400 });
  const model = chatModel();
  if (!model) return Response.json({ error: "no model is set up" }, { status: 503 });

  const [view, claims, items] = await Promise.all([learnView(user.id), listClaims(user.id).catch((): ClaimOut[] | null => null), listItems(user.id).catch((): LearnItem[] => [])]);
  const summary = learnSummary({
    books: view.books.map((b) => ({ title: b.title, status: b.status })),
    subjects: [...new Set(view.books.flatMap((b) => b.subjects.map((s) => s.name)).concat(view.known.map((s) => s.name)))],
    claims: (claims ?? []).map((c) => ({ text: c.text, stance: c.stance?.stance ?? null })),
    items: items.map((i) => ({ title: i.title, medium: mediumOf(i.medium).label.toLowerCase(), status: i.status })),
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
          show_ways: tool({
            description: "Show what the person learns from by the ways of learning: books and items (lectures, debates, films, music, games) under teaching, dialectic, imitation, practice, and rest and leisure.",
            inputSchema: z.object({}),
            execute: async () => {
              const wayOf = (b: (typeof view.books)[number]) => b.way ?? bookWay(b.catalog ? categoryOf(b.catalog) : null, b.catalog?.subjects ?? []);
              const ways = [...WAYS, LEISURE].map((w) => ({
                id: w.label.slice(0, 1), name: w.label,
                books: [...view.books.filter((b) => w.id !== "leisure" && wayOf(b) === w.id).map((b) => b.title), ...items.filter((i) => placeOf(i) === w.id).map((i) => `${i.title} (${mediumOf(i.medium).label.toLowerCase()})`)],
                subjects: [] as string[],
              })).filter((w) => w.books.length);
              show({ kind: "shelves", shelves: ways });
              return ways.length ? `Shown: ${ways.map((w) => `${w.name}: ${w.books.join(", ")}`).join("; ")}` : "Nothing on their shelves yet.";
            },
          }),
          find_item: tool({
            description: "Look up a lecture, debate, podcast, film, series, play, music, art or game for the person to add: a link, or a search of the free catalog for its medium.",
            inputSchema: z.object({ medium: z.enum(MEDIA.map((m) => m.id) as [string, ...string[]]), query: z.string().min(1).max(300).describe("A link, or words of its title") }),
            execute: async ({ medium, query }) => {
              const r = /^https?:\/\//i.test(query.trim()) ? await lookLink(query.trim(), catalogFetch).then((x) => ({ found: x.found ? [x.found] : [], error: x.error })) : await searchMedium(medium, query, catalogFetch);
              if (!r.found.length) {
                show({ kind: "note", text: r.error ?? `Nothing found for “${query}”.` });
                return `Nothing found${r.error ? `: ${r.error}` : ""}. Do not name one from memory.`;
              }
              const have = new Set(items.map((i) => `${i.medium}|${i.title.toLowerCase()}`));
              for (const f of r.found.slice(0, 5)) show({ kind: "found_item", medium: f.medium, label: mediumOf(f.medium).label, title: f.title, ...(f.by ? { by: f.by } : {}), ...(f.image ? { image: f.image } : {}), source: f.source, kept: have.has(`${f.medium}|${f.title.toLowerCase()}`), item: JSON.stringify(f) });
              return `Shown: ${r.found.slice(0, 5).map((f) => `${f.title}${f.by ? ` (${f.by})` : ""} from ${f.source}`).join("; ")}. Not kept unless the person taps Add.`;
            },
          }),
          quote_transcript: tool({
            description: "Find words in the transcripts the person pasted for their videos and recordings, and show each place with its moment.",
            inputSchema: z.object({ words: z.string().min(2).max(100), item: z.string().max(200).optional().describe("Words of the item's title, when the person names one") }),
            execute: async ({ words, item }) => {
              const withText = items.filter((i) => i.transcript && (!item || i.title.toLowerCase().includes(item.toLowerCase())));
              const out: string[] = [];
              for (const i of withText.slice(0, 10)) {
                const full = await getItem(user.id, i.id).catch(() => null);
                if (!full?.transcriptText) continue;
                for (const part of searchTranscript(readTranscript(full.transcriptText), words, 3)) {
                  const cite = `${[i.by, i.title].filter(Boolean).join(", ")}, at ${clock(part.at)}`;
                  show({ kind: "quote", text: part.text, cite, href: `/learn/items/${i.id}` });
                  out.push(`"${part.text}" (${cite})`);
                  if (out.length >= 5) break;
                }
                if (out.length >= 5) break;
              }
              if (!out.length) {
                show({ kind: "note", text: `“${words}” was not found in your transcripts.` });
                return "Not found. Do not quote from memory; say it was not found.";
              }
              return `Shown: ${out.join("; ")}`;
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
          quote_book: tool({
            description: "Find words in the text of a book the person reads in Reflect (one they are reading, or the one named) and show each place with its chapter, to quote and cite.",
            inputSchema: z.object({ words: z.string().min(2).max(100).describe("A few words that occur in the book, as they would be written in it"), book: z.string().max(200).optional().describe("Words of the book's title, when the person names one") }),
            execute: async ({ words, book }) => {
              const r = await quoteFromShelves(view.books, words, book).catch(() => null);
              if (!r) {
                show({ kind: "note", text: book ? `“${words}” was not found in ${book} in Reflect.` : `“${words}” was not found in the books you are reading in Reflect.` });
                return "Not found. Do not quote from memory; say it was not found.";
              }
              const cites = r.hits.map((h, i) => {
                const text = `${h.before}${h.match}${h.after}`.trim();
                const cite = [r.author, r.title, r.chapters[i]].filter(Boolean).join(", ");
                show({ kind: "quote", text, cite, href: `/learn/works/${r.work}/read?c=${h.chapter}&find=${encodeURIComponent(words)}&p=${h.p}` });
                return `"${text}" (${cite})`;
              });
              return `Shown, ${r.total} ${r.total === 1 ? "place" : "places"} in all: ${cites.join("; ")}`;
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
