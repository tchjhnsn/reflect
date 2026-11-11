// What a person's own AI can do with their Reflect journal, over MCP. Each tool is a plain function of the
// signed-in person, so the same calls are tested without HTTP. The AI proposes; the person decides: a
// reading is kept only when the person has said yes to it in the conversation, and nothing here guesses.
import { McpServer } from "@modelcontextprotocol/server";
import { z } from "zod";
import { acceptLocalDate, type Entry } from "@reflect/core/journal";
import type { IsoDate } from "@reflect/core/person";
import { closeEntry, createEntry, entryText, listEntries } from "./journal-store";
import { keepPlan, legacyStatus, type PlanInput } from "./legacy-store";
import { addBook, addSubjects, getBook, keepRecall, learnView, rateSubject, setBookStatus } from "./learn-store";
import { addCourses, addCredential, listCourses, listCredentials } from "./learn-sources-store";
import { BOOK_STATUS, GRASP, type BookStatus } from "@reflect/core/learning";
import { lookupIsbn, type Fetch } from "@reflect/core/openlibrary";
import { siteUrl } from "./site";
import { STANCES, type Stance, type Target } from "@reflect/core/ideas";
import { FILM_KINDS, FILM_STATUS, RESOURCE_KINDS, type FilmStatus } from "@reflect/core/learning";
import { addInput, keepIdea, listIdeas, listInputs, markSubject, resolveByTitle, inputTitle } from "./ideas-store";

export const GUIDE = [
  "Reflect keeps a person's journal, what they learn, their own ideas and the plan for what survives them. You help them write and keep things; you never decide for them.",
  "",
  "Rules: save only what the person asked you to save, in their words. Never infer facts about the person beyond what they wrote, and never share their entries.",
].join("\n");

const text = (v: unknown) => ({ content: [{ type: "text" as const, text: typeof v === "string" ? v : JSON.stringify(v, null, 2) }] });

const entryOut = (e: Entry) => ({ id: e.id, date: e.localDate, closed: e.closedAt !== null, text: e.parts.map((p) => p.text).join("\n\n") });

export async function toolListEntries(userId: string, limit: number) {
  return (await listEntries(userId, limit)).map(entryOut);
}

export async function toolWriteEntry(userId: string, entryTextIn: string, localDate: string, now = new Date()) {
  const body = entryTextIn.trim();
  if (!body) throw new Error("an entry needs words");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(localDate) || !acceptLocalDate(localDate as IsoDate, now)) throw new Error("the date must be the person's today (YYYY-MM-DD)");
  const id = await createEntry(userId, body, localDate as IsoDate, null);
  await closeEntry(userId, id);
  return { id };
}

export async function toolGetEntry(userId: string, entryId: string): Promise<{ id: string; text: string }> {
  const t = await entryText(userId, entryId);
  if (t === null) throw new Error("no such entry");
  return { id: entryId, text: t };
}

/** Keep the person's legacy release plan, as they stated it. Returns each trustee's link for the person to hand over. */
export async function toolKeepLegacyPlan(userId: string, plan: PlanInput) {
  const r = await keepPlan(userId, plan);
  return { trustees: r.trustees.map((t) => ({ name: t.name, link: `${siteUrl()}/legacy/trustee/${t.token}` })), gifts: r.gifts, see: `${siteUrl()}/legacy` };
}

export async function toolLegacyStatus(userId: string) {
  const s = await legacyStatus(userId);
  return { plan: s.plan, trustees: s.trustees.map((t) => ({ name: t.name, contact: t.contact })), gifts: s.gifts, release: s.release };
}

const openLibrary: Fetch = (url) => fetch(url, { headers: { "user-agent": "reflect-app" } });

/** Keep a book the person names. With an ISBN, Open Library fills the authors; its subjects come back as suggestions only. */
export async function toolAddBook(userId: string, b: { title: string; isbn?: string; status?: BookStatus; subjects?: string[] }, fetcher: Fetch = openLibrary) {
  const catalog = b.isbn ? await lookupIsbn(b.isbn, fetcher).catch(() => null) : null;
  const id = await addBook(userId, { ...b, authors: catalog?.authors ?? [], catalog });
  return { book_id: id, authors: catalog?.authors ?? [], suggested_subjects: catalog?.subjects ?? [], note: "Suggested subjects are Open Library's. Add only the ones the person says fit, with add_book_subjects." };
}

/** Keep what the person named of a book without looking, then the book's subjects to rate. */
export async function toolRecallCheck(userId: string, bookId: string, recalled: string) {
  const r = await keepRecall(userId, bookId, recalled);
  const book = await getBook(userId, bookId);
  return { ...r, contents: book?.catalog?.contents ?? [], grasp_levels: GRASP, next: "Show the person the subjects they named and the rest, then ask their grasp of each in their words and keep it with rate_subject. Do not rate for them." };
}

export async function toolLearning(userId: string) {
  const v = await learnView(userId);
  const [courses, credentials, inputs] = await Promise.all([listCourses(userId), listCredentials(userId), listInputs(userId)]);
  return {
    shown_not_rated: v.shown.map((s) => s.name),
    suggested_not_known: v.suggested.map((s) => s.name),
    inputs: inputs.map((i) => ({ id: i.id, kind: i.body.kind, title: inputTitle(i.body), subjects: i.subjects })),
    known: v.known.map((s) => ({ subject: s.name, grasp: s.grasp, vouched_for_by: s.vouches })),
    vouched_not_rated: v.vouched.map((s) => ({ subject: s.name, vouched_for_by: s.vouches })),
    courses: courses.map((c) => ({ ...c.row, your_grasp: c.subject.grasp?.grasp ?? null })),
    credentials: credentials.map((c) => ({ name: c.name, issuer: c.issuer, issued: c.issued ?? null, subjects: c.subjects, checked: c.checked ?? null })),
    books: v.books.map((b) => ({ id: b.id, title: b.title, authors: b.authors, status: b.status, finished: b.finishedOn, subjects: b.subjects.map((s) => s.name), not_yet_rated: b.unrated })),
  };
}

const legacySchema = z.object({
  confirmations: z.number().int().min(1).describe("how many trustees must confirm the death"),
  wait_days: z.number().int().min(0).max(365).describe("days after enough confirmations in which the person or any trustee can stop the release"),
  trustees: z.array(z.object({ name: z.string().min(1), contact: z.string().min(1) })).min(1),
  gifts: z.array(z.object({
    to: z.string().min(1),
    contact: z.string().min(1),
    items: z.array(z.object({ title: z.string().min(1), text: z.string() })).min(1).describe("only letters and writing the person made to survive them, word for word"),
  })),
});

/** One MCP server for one signed-in person. */
export function reflectServer(userId: string): McpServer {
  const server = new McpServer({ name: "reflect", version: "0.1.0" }, { instructions: GUIDE });

  server.registerTool("reflect_guide", {
    title: "How Reflect works",
    description: "What Reflect keeps and the rules for an AI working in it.",
    annotations: { readOnlyHint: true },
  }, async () => text(GUIDE));

  server.registerTool("list_entries", {
    title: "List journal entries",
    description: "The person's journal entries, newest first.",
    inputSchema: z.object({ limit: z.number().int().min(1).max(200).default(20) }),
    annotations: { readOnlyHint: true },
  }, async ({ limit }) => text(await toolListEntries(userId, limit)));

  server.registerTool("write_entry", {
    title: "Write a journal entry",
    description: "Save an entry in the person's own words. Write only what they asked you to save, as they said it; never compose it for them.",
    inputSchema: z.object({ text: z.string().min(1), local_date: z.string().describe("the person's today, YYYY-MM-DD") }),
  }, async ({ text: t, local_date }) => text(await toolWriteEntry(userId, t, local_date)));

  server.registerTool("get_entry", {
    title: "Read one entry",
    description: "One entry's text.",
    inputSchema: z.object({ entry_id: z.string().uuid() }),
    annotations: { readOnlyHint: true },
  }, async ({ entry_id }) => text(await toolGetEntry(userId, entry_id)));

  server.registerTool("keep_legacy_plan", {
    title: "Keep the person's legacy release plan",
    description: "Send the release plan the person wrote (in their vault, legacy/release.json and what it gives) to Reflect, which carries it out after their death: trustees confirm, a wait passes in which they or any trustee can stop it, then each person receives their part. Send only the legacy letters and entries marked to survive them, word for word, and only after the person says yes. A new plan replaces the old one and revokes its links. Give the person the trustee links it returns.",
    inputSchema: legacySchema,
  }, async ({ confirmations, wait_days, trustees, gifts }) => text(await toolKeepLegacyPlan(userId, { confirmations, waitDays: wait_days, trustees, gifts })));

  server.registerTool("legacy_status", {
    title: "The legacy release plan and any release",
    description: "Who the trustees are, who receives something (not what), and whether a release is under way.",
    annotations: { readOnlyHint: true },
  }, async () => text(await toolLegacyStatus(userId)));

  server.registerTool("learning", {
    title: "What the person knows",
    description: "What the person took in: subjects they rated as known (could teach, explain or remember the gist), subjects shown but not rated, subjects suggested, their books (with subjects of read books not yet rated), films, talks, conversations, links, courses and credentials. Nothing is known until the person rates it. Their own ideas are under ideas.",
    annotations: { readOnlyHint: true },
  }, async () => text(await toolLearning(userId)));

  server.registerTool("add_book", {
    title: "Keep a book",
    description: "Keep a book the person names, where they are with it, and the key subjects as they name them. Give an ISBN to look it up in Open Library; its subjects come back as suggestions, never kept without the person.",
    inputSchema: z.object({ title: z.string().min(1), isbn: z.string().optional(), status: z.enum(BOOK_STATUS as unknown as [string, ...string[]]).default("read"), subjects: z.array(z.string().min(1)).max(60).optional() }),
  }, async ({ title, isbn, status, subjects }) => text(await toolAddBook(userId, { title, ...(isbn ? { isbn } : {}), status: status as BookStatus, ...(subjects ? { subjects } : {}) })));

  server.registerTool("add_book_subjects", {
    title: "Add subjects to a book",
    description: "Add subjects the person says the book covers.",
    inputSchema: z.object({ book_id: z.string().uuid(), subjects: z.array(z.string().min(1)).min(1).max(60) }),
  }, async ({ book_id, subjects }) => text(await addSubjects(userId, book_id, subjects).then(() => ({ ok: true }))));

  server.registerTool("set_book_status", {
    title: "Where the person is with a book",
    description: "want, reading, read, abandoned or referenced (cited or mentioned, how far read not said). Reading records when they started; read records when they finished.",
    inputSchema: z.object({ book_id: z.string().uuid(), status: z.enum(BOOK_STATUS as unknown as [string, ...string[]]) }),
  }, async ({ book_id, status }) => text(await setBookStatus(userId, book_id, status as BookStatus).then(() => ({ ok: true }))));

  server.registerTool("recall_check", {
    title: "What the person remembers of a book",
    description: "First ask the person, without showing them anything, to name the book's key subjects from memory. Send their words here as they said them. It keeps the recall and returns which of the book's subjects they named, the rest, and the book's contents if known.",
    inputSchema: z.object({ book_id: z.string().uuid(), recalled: z.string().min(1).describe("the person's own words, before seeing any list") }),
  }, async ({ book_id, recalled }) => text(await toolRecallCheck(userId, book_id, recalled)));

  server.registerTool("add_courses", {
    title: "Keep courses from the person's transcript",
    description: "Courses read from a transcript the person shared with you. Copy each row exactly as the transcript writes it, show the rows to the person, and send only the ones they confirmed. The grade is the school's; ask the person their own grasp of each and keep it with rate_subject (the course title is the subject).",
    inputSchema: z.object({ institution: z.string().optional(), courses: z.array(z.object({ term: z.string(), code: z.string().optional(), title: z.string().min(1), credits: z.number().min(0).max(99).optional(), grade: z.string().optional() })).min(1).max(200) }),
  }, async ({ institution, courses }) => text({ kept: await addCourses(userId, courses.map((c) => ({ ...c, ...(institution ? { institution } : {}) }))) }));

  server.registerTool("add_credential", {
    title: "Keep a credential",
    description: "A certificate, license or badge the person holds, as they state it, with the subjects it covers.",
    inputSchema: z.object({ name: z.string().min(1), issuer: z.string().min(1), issued: z.string().optional().describe("YYYY, YYYY-MM or YYYY-MM-DD"), expires: z.string().optional(), url: z.string().url().optional(), subjects: z.array(z.string().min(1)).max(60).optional() }),
  }, async ({ name, ...c }) => text({ credential_id: await addCredential(userId, name, c) }));

  server.registerTool("rate_subject", {
    title: "Keep the person's grasp of a subject",
    description: "The person's own rating, never yours: teach (could teach it), explain (could explain the key ideas), gist (remember the gist) or forgot (don't remember it). A new rating never changes an old one; the latest counts.",
    inputSchema: z.object({ subject: z.string().min(1), grasp: z.enum(["teach", "explain", "gist", "forgot"]), book_id: z.string().uuid().optional(), note: z.string().optional() }),
  }, async ({ subject, grasp, book_id, note }) => text(await rateSubject(userId, subject, grasp, { ...(book_id ? { bookId: book_id } : {}), ...(note ? { note } : {}) }).then(() => ({ ok: true }))));

  server.registerTool("add_input", {
    title: "Keep a film, talk, conversation or link",
    description: "Something the person took in beside books: a film, documentary or series; a talk (someone they saw speak); a conversation (with whom, what about); or a link (a post or thread on X, Substack or Reddit, a video, an article). Keep only what the person confirms, as they state it.",
    inputSchema: z.object({
      kind: z.enum(["film", "talk", "conversation", "link"]),
      title: z.string().optional().describe("the title; for a conversation, what it was about"),
      who: z.array(z.string().min(1)).optional().describe("the speaker, the people in the conversation, or the author"),
      url: z.string().url().optional(),
      on: z.string().optional().describe("YYYY, YYYY-MM or YYYY-MM-DD"),
      film: z.enum(FILM_KINDS as unknown as [string, ...string[]]).optional(),
      status: z.enum(FILM_STATUS as unknown as [string, ...string[]]).optional(),
      year: z.number().int().optional(),
      link: z.enum(RESOURCE_KINDS as unknown as [string, ...string[]]).optional(),
      subjects: z.array(z.string().min(1)).max(60).optional(),
    }),
  }, async (a) => {
    const title = a.title ?? "";
    const who = a.who ?? [];
    const body = a.kind === "film" ? { kind: "film" as const, title, film: a.film ?? "film", ...(a.year ? { year: a.year } : {}), status: (a.status ?? "watched") as FilmStatus }
      : a.kind === "talk" ? { kind: "talk" as const, title, speaker: who.join(", "), ...(a.on ? { on: a.on } : {}), ...(a.url ? { url: a.url } : {}) }
      : a.kind === "conversation" ? { kind: "conversation" as const, with: who, on: a.on ?? new Date().toISOString().slice(0, 10), about: title }
      : { kind: "link" as const, title: title || a.url || "", link: a.link ?? "article", ...(a.url ? { url: a.url } : {}), ...(who.length ? { by: who.join(", ") } : {}) };
    return text({ input_id: await addInput(userId, body, a.subjects ?? []) });
  });

  server.registerTool("mark_subject", {
    title: "Note that the person showed knowledge of a subject, or suggest one",
    description: "shown: you saw the person display knowledge of an established subject (they explained it, applied it, corrected you, taught it); say what you saw. suggested: you think a subject relates to them. Neither is a grasp; only the person rates their grasp (rate_subject). A framing of the person's own is an idea (keep_idea), not a subject. Mark only with the person's yes.",
    inputSchema: z.object({ subject: z.string().min(1), kind: z.enum(["shown", "suggested"]), by: z.string().min(1).describe("your product name"), evidence: z.string().optional() }),
  }, async ({ subject, kind, by, evidence }) => text(await markSubject(userId, subject, kind, by, evidence).then(() => ({ ok: true }))));

  server.registerTool("ideas", {
    title: "The person's own ideas",
    description: "Each idea in the person's latest words, what it came from (and maybe from), and recommendations placed from supports to opposes. Ideas are outputs; what the person took in is under learning.",
    annotations: { readOnlyHint: true },
  }, async () => text((await listIdeas(userId)).map(({ labels, ...i }) => {
    const named = (t: Target) => ({ ...(t.ref ? { ref: t.ref, title: labels[t.ref] } : { text: t.text }), ...(t.url ? { url: t.url } : {}) });
    return { id: i.id, title: i.title, statement: i.statement, from: i.from.map((x) => ({ ...named(x), how: x.how })), maybe_from: i.maybe.map((x) => ({ ...named(x), how: x.how })), recommendations: i.related.map((r) => ({ ...named(r), stance: r.stance, by: r.by, note: r.note })) };
  })));

  server.registerTool("keep_idea", {
    title: "Keep one of the person's ideas",
    description: `An idea of the person's own: a framing, thesis or hybrid ("X applied to Y"). Say it in their words. Name inputs by title (a book, film, talk, conversation or link they kept, matched by title, else kept as their words): from when the person says it fed the idea, maybe_from when they are not sure. Recommendations are things to study, not inputs, each with a stance: ${STANCES.map((x) => `${x.id} (${x.text})`).join(", ")}. A new line each time; the latest line about a thing decides where it stands. Send only what the person said yes to.`,
    inputSchema: z.object({
      title: z.string().min(1).max(200),
      statement: z.string().optional().describe("required for a new idea; a restatement otherwise"),
      from: z.array(z.object({ title: z.string().min(1), how: z.string().optional() })).max(50).optional(),
      maybe_from: z.array(z.object({ title: z.string().min(1), how: z.string().optional() })).max(50).optional(),
      recommendations: z.array(z.object({ title: z.string().min(1), url: z.string().url().optional(), stance: z.enum(STANCES.map((x) => x.id) as [Stance, ...Stance[]]), by: z.string().min(1), note: z.string().optional() })).max(50).optional(),
    }),
  }, async ({ title, statement, from, maybe_from, recommendations }) => {
    const t = async (x: { title: string; url?: string }) => ({ ...(await resolveByTitle(userId, x.title)), ...(x.url ? { url: x.url } : {}) });
    const id = await keepIdea(userId, title, {
      ...(statement ? { statement } : {}),
      from: await Promise.all((from ?? []).map(async (x) => ({ target: await t(x), ...(x.how ? { how: x.how } : {}) }))),
      maybe: await Promise.all((maybe_from ?? []).map(async (x) => ({ target: await t(x), ...(x.how ? { how: x.how } : {}) }))),
      related: await Promise.all((recommendations ?? []).map(async (r) => ({ target: await t(r), stance: r.stance, by: r.by, ...(r.note ? { note: r.note } : {}) }))),
    });
    return text({ idea_id: id });
  });

  return server;
}
