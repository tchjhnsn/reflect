// What a person's own AI can do with their Reflect journal, over MCP. Each tool is a plain function of the
// signed-in person, so the same calls are tested without HTTP. The AI proposes; the person decides: a
// reading is kept only when the person has said yes to it in the conversation, and nothing here guesses.
import { McpServer } from "@modelcontextprotocol/server";
import { z } from "zod";
import { KINDS, ROLES, SORTS, presentMeaning, type Binding, type Concept, type Reading } from "@reflect/core/bindings";
import { acceptLocalDate, type Entry } from "@reflect/core/journal";
import type { IsoDate } from "@reflect/core/person";
import { SALIENCE_PROMPT } from "@reflect/core/salience-prompt";
import { PRIMES } from "@reflect/core/state";
import { conceptHistory, keepReading, onThisConcept, readings } from "./binding-store";
import { closeEntry, createEntry, entryText, listEntries } from "./journal-store";
import { keepPlan, legacyStatus, type PlanInput } from "./legacy-store";
import { addBook, addSubjects, getBook, keepRecall, learnView, rateSubject, setBookStatus } from "./learn-store";
import { addCourses, addCredential, listCourses, listCredentials } from "./learn-sources-store";
import { BOOK_STATUS, GRASP, type BookStatus } from "@reflect/core/learning";
import { lookupIsbn, type Fetch } from "@reflect/core/openlibrary";
import { siteUrl } from "./site";

/** The elements, from the reader's own description, so an outside AI uses the same words. */
const ELEMENTS = SALIENCE_PROMPT.slice(SALIENCE_PROMPT.indexOf("The 16 elements"), SALIENCE_PROMPT.indexOf("\n\nRules")).trim();

export const GUIDE = [
  "Reflect keeps a person's journal and what they find in it. You help them write and reflect; you never decide for them.",
  "",
  ELEMENTS,
  "",
  "A binding ties one element, with a weight from 0 to 1, to one or more concepts the entry names:",
  `- kind: ${KINDS.join(", ")} (entity is a noun, event a verb, quality an adjective or adverb, time a when)`,
  `- sort, for an entity only: ${SORTS.join(", ")}`,
  `- role, when it matters: ${ROLES.join(", ")}`,
  "- words: the entry's own words for the concept (required for every kind but time).",
  "",
  "Rules: propose bindings as questions, show them to the person, and call keep_reading only with what they said yes to, corrected as they said.",
  "A kept reading is never changed. Looking back later is a new reading of the same entry, and the first stays as it was felt then.",
  "Never infer facts about the person beyond what they wrote, and never share their entries.",
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

export async function toolGetEntry(userId: string, entryId: string): Promise<{ id: string; text: string; readings: Reading[] }> {
  const t = await entryText(userId, entryId);
  if (t === null) throw new Error("no such entry");
  return { id: entryId, text: t, readings: await readings(userId, entryId) };
}

export async function toolKeepReading(userId: string, entryId: string, bindings: Binding[]) {
  return { readAt: await keepReading(userId, entryId, bindings) };
}

export async function toolConceptHistory(userId: string, concept: Concept, now = new Date()) {
  const points = await conceptHistory(userId, concept);
  return { entries: await onThisConcept(userId, concept), points, now: presentMeaning(points, now.toISOString()) };
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
  const [courses, credentials] = await Promise.all([listCourses(userId), listCredentials(userId)]);
  return {
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

const conceptSchema = z.object({
  kind: z.enum(KINDS as [string, ...string[]]).describe("entity (noun), event (verb), quality (adjective or adverb), time"),
  name: z.string().min(1).describe("the concept in the person's words, e.g. Dad, leave, quiet"),
  sort: z.enum(SORTS as [string, ...string[]]).optional().describe("for an entity only"),
});
const bindingSchema = z.object({
  element: z.enum(PRIMES as [string, ...string[]]),
  weight: z.number().min(0).max(1),
  on: z.array(z.object({
    concept: conceptSchema,
    words: z.string().optional().describe("the entry's own words for it; required except for a time"),
    role: z.enum(ROLES as [string, ...string[]]).optional(),
  })).min(1),
});

/** One MCP server for one signed-in person. */
export function reflectServer(userId: string): McpServer {
  const server = new McpServer({ name: "reflect", version: "0.1.0" }, { instructions: GUIDE });

  server.registerTool("reflect_guide", {
    title: "How Reflect works",
    description: "The 16 elements, how a binding is written, and the rules. Read this before proposing a reading.",
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
    description: "An entry's text and every reading kept for it, oldest first. The first reading is the entry as felt then.",
    inputSchema: z.object({ entry_id: z.string().uuid() }),
    annotations: { readOnlyHint: true },
  }, async ({ entry_id }) => text(await toolGetEntry(userId, entry_id)));

  server.registerTool("keep_reading", {
    title: "Keep a reading the person approved",
    description: "Keep the bindings the person said yes to for one entry. Call it only after showing them each binding and hearing yes. A kept reading is never changed.",
    inputSchema: z.object({ entry_id: z.string().uuid(), bindings: z.array(bindingSchema).min(1) }),
  }, async ({ entry_id, bindings }) => text(await toolKeepReading(userId, entry_id, bindings as Binding[])));

  server.registerTool("concept_history", {
    title: "What a concept has carried",
    description: "Every binding of one concept across the person's entries, and what it carries now (later readings count more, earlier ones never drop out).",
    inputSchema: z.object({ concept: conceptSchema }),
    annotations: { readOnlyHint: true },
  }, async ({ concept }) => text(await toolConceptHistory(userId, concept as Concept)));

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
    description: "Subjects the person rated as known (could teach, explain or remember the gist), and their books with the subjects of read books not yet rated. Nothing is known until the person rates it.",
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
    description: "want, reading, read or abandoned. Reading records when they started; read records when they finished.",
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

  return server;
}
