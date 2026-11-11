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

  return server;
}
