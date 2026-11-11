import { randomUUID } from "node:crypto";
import { foldIdea, isStance, STANCES, type Idea, type IdeaLine, type Stance, type Target } from "@reflect/core/ideas";
import { FILM_KINDS, FILM_STATUS, RESOURCE_KINDS, sourceOf, type FilmStatus } from "@reflect/core/learning";
import { db } from "@/db/pool";
import { open, seal, tag } from "./crypto";
import { subjectId, subjects, tx } from "./learn-store";

/**
 * Inputs apart from outputs (Torian, 2026-11-11). Films, talks, conversations and links are inputs beside
 * books; marks are evidence beside a subject that is never a grasp (shown, suggested); an idea is the
 * person's own output, a run of lines that say it, link what it came from or may have, place
 * recommendations from supports to opposes, or drop one. The latest line about a thing decides where it
 * stands, folded by the same code as the files host.
 */

export type InputKind = "film" | "talk" | "conversation" | "link";
export type InputBody =
  | { kind: "film"; title: string; film: string; year?: number; by?: string; status: FilmStatus }
  | { kind: "talk"; title: string; speaker: string; on?: string; where?: string; url?: string }
  | { kind: "conversation"; with: string[]; on: string; about: string }
  | { kind: "link"; title: string; link: string; url?: string; source?: string; by?: string };
export type InputOut = { id: string; body: InputBody; subjects: string[]; source?: string };

const DAY = /^\d{4}(-\d{2}(-\d{2})?)?$/;
const one = (s: string) => s.replace(/\s+/g, " ").trim();
export const inputTitle = (b: InputBody) => (b.kind === "conversation" ? `With ${b.with.join(", ")}: ${b.about}` : b.title);

function check(b: InputBody): InputBody {
  if (b.kind === "film") {
    if (!one(b.title)) throw new Error("a film needs a title");
    if (!(FILM_KINDS as readonly string[]).includes(b.film)) throw new Error(`a film is one of ${FILM_KINDS.join(", ")}`);
    if (!(FILM_STATUS as readonly string[]).includes(b.status)) throw new Error(`status is one of ${FILM_STATUS.join(", ")}`);
    if (b.year !== undefined && !(Number.isInteger(b.year) && b.year > 1870 && b.year < 2200)) throw new Error("the year is a year like 1957");
    return { ...b, title: one(b.title) };
  }
  if (b.kind === "talk") {
    if (!one(b.title) || !one(b.speaker)) throw new Error("a talk has a title and who spoke");
    if (b.on && !DAY.test(b.on)) throw new Error("dates are YYYY, YYYY-MM or YYYY-MM-DD");
    if (b.url && !/^https?:\/\/\S+$/.test(b.url)) throw new Error("the link must start with http");
    return { ...b, title: one(b.title), speaker: one(b.speaker) };
  }
  if (b.kind === "conversation") {
    const who = b.with.map(one).filter(Boolean);
    if (!who.length || !one(b.about)) throw new Error("a conversation says who it was with and what about");
    if (!DAY.test(b.on)) throw new Error("dates are YYYY, YYYY-MM or YYYY-MM-DD");
    return { ...b, with: who, about: one(b.about) };
  }
  if (!one(b.title)) throw new Error("a link needs a title");
  if (!(RESOURCE_KINDS as readonly string[]).includes(b.link)) throw new Error(`a link is one of ${RESOURCE_KINDS.join(", ")}`);
  if (b.url && !/^https?:\/\/\S+$/.test(b.url)) throw new Error("the link must start with http");
  const source = b.url ? sourceOf(b.url) : undefined;
  return { ...b, title: one(b.title), ...(source ? { source } : {}) };
}

const dedupe = (b: InputBody) => (b.kind === "conversation" ? `conversation|${b.on}|${b.with.join(";").toLowerCase()}|${b.about.toLowerCase()}` : b.kind === "link" && b.url ? `link|${b.url}` : `${b.kind}|${b.title.toLowerCase()}`);

/** Keep an input, once (by title, link, or for a conversation by day, people and topic). Returns its id. */
export async function addInput(userId: string, input: InputBody, subjectNames: string[] = [], source?: string): Promise<string> {
  const b = check(input);
  return tx(async (c) => {
    const t = tag(userId, dedupe(b));
    const id = randomUUID();
    await c.query("INSERT INTO learn_input (id, user_id, kind, dedupe_tag, body_ct) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (user_id, dedupe_tag) DO NOTHING", [id, userId, b.kind, t, seal(userId, JSON.stringify({ ...b, ...(source ? { from: source } : {}) }))]);
    const kept = (await c.query("SELECT id FROM learn_input WHERE user_id = $1 AND dedupe_tag = $2", [userId, t])).rows[0]!.id as string;
    for (const s of subjectNames) if (one(s)) await c.query("INSERT INTO learn_input_subject (input_id, subject_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [kept, await subjectId(c, userId, s)]);
    return kept;
  });
}

export async function listInputs(userId: string): Promise<InputOut[]> {
  const subs = await subjects(userId);
  const r = await db().query("SELECT id, body_ct FROM learn_input WHERE user_id = $1 ORDER BY created_at", [userId]);
  const links = await db().query("SELECT s.input_id, s.subject_id FROM learn_input_subject s JOIN learn_input i ON i.id = s.input_id WHERE i.user_id = $1", [userId]);
  return r.rows.map((x) => {
    const { from, ...body } = JSON.parse(open(userId, x.body_ct as string)) as InputBody & { from?: string };
    return { id: x.id as string, body: body as InputBody, subjects: links.rows.filter((l) => l.input_id === x.id).map((l) => subs.get(l.subject_id as string)?.name ?? "").filter(Boolean), ...(from ? { source: from } : {}) };
  });
}

/** Evidence beside a subject that is never a grasp: someone saw the person show it, or suggests it. */
export async function markSubject(userId: string, name: string, kind: "shown" | "suggested", by: string, text?: string): Promise<void> {
  if (!one(by)) throw new Error("say who saw it or suggested it");
  if (kind === "shown" && !one(text ?? "")) throw new Error("say what they saw");
  await tx(async (c) => {
    await c.query("INSERT INTO learn_subject_mark (id, user_id, subject_id, kind, body_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, await subjectId(c, userId, name), kind, seal(userId, JSON.stringify({ by: one(by), ...(text ? { text: one(text) } : {}) }))]);
  });
}

/** What can be named as an input or a recommendation: a book, an input, a subject or another idea, by id. */
export type RefKind = "book" | "input" | "subject" | "idea";
const REF = /^(book|input|subject|idea):([0-9a-f-]{36})$/;

export type IdeaChange = {
  statement?: string;
  from?: { target: Target; how?: string }[];
  maybe?: { target: Target; how?: string }[];
  related?: { target: Target; stance: Stance; by: string; note?: string }[];
  drop?: Target[];
};
export type IdeaOut = Idea & { labels: Record<string, string> };

async function ownsRef(userId: string, ref: string): Promise<boolean> {
  const m = REF.exec(ref);
  if (!m) return false;
  const table = { book: "learn_book", input: "learn_input", subject: "learn_subject", idea: "idea" }[m[1] as RefKind];
  return (await db().query(`SELECT 1 FROM ${table} WHERE id = $1 AND user_id = $2`, [m[2], userId])).rowCount === 1;
}

/** Keep an idea: say or restate it, link inputs, add recommendations, drop any. A new line each time. */
export async function keepIdea(userId: string, title: string, change: IdeaChange): Promise<string> {
  const name = one(title);
  if (!name || name.length > 200) throw new Error("an idea has a short name");
  const day = new Date().toISOString().slice(0, 10);
  const lines: IdeaLine[] = [];
  const target = async (t: Target) => {
    if (t.ref && !(await ownsRef(userId, t.ref))) throw new Error("that input is not yours or does not exist");
    if (!t.ref && !one(t.text ?? "")) throw new Error("name the input or recommendation");
    if (t.url && !/^https?:\/\/\S+$/.test(t.url)) throw new Error("the link must start with http");
  };
  if (change.statement?.trim()) {
    if (change.statement.length > 20_000) throw new Error("the statement is too long");
    lines.push({ kind: "stated", on: day, text: change.statement.trim() });
  }
  for (const [kind, xs] of [["from", change.from], ["maybe", change.maybe]] as const) for (const x of xs ?? []) {
    await target(x.target);
    lines.push({ kind, input: { on: day, ...x.target, ...(x.how?.trim() ? { how: x.how.trim() } : {}) } });
  }
  for (const r of change.related ?? []) {
    if (!isStance(r.stance)) throw new Error(`a recommendation's stance is one of ${STANCES.map((s) => s.id).join(", ")}`);
    if (!one(r.by)) throw new Error("say who suggested it");
    await target(r.target);
    lines.push({ kind: "related", rec: { on: day, stance: r.stance, by: one(r.by), ...r.target, ...(r.note?.trim() ? { note: r.note.trim() } : {}) } });
  }
  for (const t of change.drop ?? []) lines.push({ kind: "dropped", on: day, target: t });
  return tx(async (c) => {
    const t = tag(userId, `idea|${name.toLowerCase()}`);
    const found = (await c.query("SELECT id FROM idea WHERE user_id = $1 AND name_tag = $2", [userId, t])).rows[0]?.id as string | undefined;
    if (!found && !lines.some((l) => l.kind === "stated")) throw new Error("a new idea needs its statement, in your own words");
    if (found && !lines.length) throw new Error("nothing to change");
    const id = found ?? randomUUID();
    if (!found) await c.query("INSERT INTO idea (id, user_id, name_tag, title_ct) VALUES ($1, $2, $3, $4)", [id, userId, t, seal(userId, name)]);
    for (const l of lines) {
      if (l.kind !== "stated" && l.kind !== "dropped") {
        const ref = l.kind === "related" ? l.rec.ref : l.input.ref;
        if (ref === `idea:${id}`) throw new Error("an idea is not its own input");
      }
      await c.query("INSERT INTO idea_line (id, user_id, idea_id, kind, body_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, id, l.kind, seal(userId, JSON.stringify(l))]);
    }
    return id;
  });
}

/** Titles for the refs an idea names, so the page can show them. */
export async function labels(userId: string): Promise<Record<string, string>> {
  const out: Record<string, string> = {};
  for (const x of (await db().query("SELECT id, title_ct FROM learn_book WHERE user_id = $1", [userId])).rows) out[`book:${x.id}`] = open(userId, x.title_ct as string);
  for (const x of (await db().query("SELECT id, title_ct FROM idea WHERE user_id = $1", [userId])).rows) out[`idea:${x.id}`] = open(userId, x.title_ct as string);
  for (const s of (await subjects(userId)).values()) out[`subject:${s.id}`] = s.name;
  for (const i of await listInputs(userId)) out[`input:${i.id}`] = inputTitle(i.body);
  return out;
}

export async function listIdeas(userId: string, only?: string): Promise<IdeaOut[]> {
  if (only && !/^[0-9a-f-]{36}$/.test(only)) return [];
  const ideas = await db().query(`SELECT id, title_ct FROM idea WHERE user_id = $1 ${only ? "AND id = $2" : ""} ORDER BY created_at DESC`, only ? [userId, only] : [userId]);
  if (!ideas.rows.length) return [];
  const lines = await db().query(`SELECT idea_id, body_ct FROM idea_line WHERE user_id = $1 ${only ? "AND idea_id = $2" : ""} ORDER BY created_at, id`, only ? [userId, only] : [userId]);
  const names = await labels(userId);
  return ideas.rows.map((x) => ({
    ...foldIdea(x.id as string, open(userId, x.title_ct as string), lines.rows.filter((l) => l.idea_id === x.id).map((l) => JSON.parse(open(userId, l.body_ct as string)) as IdeaLine)),
    labels: names,
  }));
}

export const getIdea = async (userId: string, id: string): Promise<IdeaOut | null> => (await listIdeas(userId, id))[0] ?? null;

/** Find what the person named by title among their books, inputs, subjects and ideas; else keep their words. */
export async function resolveByTitle(userId: string, title: string): Promise<Target> {
  const t = one(title).toLowerCase();
  if (/^https?:\/\/\S+$/.test(title.trim())) return { text: title.trim(), url: title.trim() };
  const names = await labels(userId);
  for (const kind of ["book", "input", "idea", "subject"] as const) {
    const hit = Object.entries(names).find(([k, v]) => k.startsWith(`${kind}:`) && v.toLowerCase() === t);
    if (hit) return { ref: hit[0] };
  }
  return { text: one(title) };
}
