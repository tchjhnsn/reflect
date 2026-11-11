// What a person knows, as files in learn/. Books are the center: a book note in learn/books/ says what the
// person read and which subjects it covers. A subject note in learn/subjects/ is a node of their knowledge
// graph, and the person rates their own grasp of it, dated, often in the context of a book: could teach
// it, could explain the key ideas, remember the gist, or don't remember it. Known is never computed: a
// subject is known only at the grasp the person last gave it, and a book read but not rated counts as read,
// not as known. A subject the person wants is aspire, or need when they name what needs it. Possible next
// is derived: the fringe of knowledge space theory (Doignon and Falmagne), a subject whose prerequisites
// are known. Subjects point at a shared map (Wikidata, Library of Congress subject headings, ESCO) instead
// of Reflect keeping one. A resource in learn/resources/ is any other link (a channel, an article, a
// course) kept for what the person wants to learn.
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { headOf } from "./lifetime.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const BOOKS = join("learn", "books");
export const SUBJECTS = join("learn", "subjects");
export const RESOURCES = join("learn", "resources");

export const WANTS = ["aspire", "need"] as const;
export type Want = (typeof WANTS)[number];

/** The person's own grasp of a subject, from most to least. Only they set it. */
export const GRASP = [
  { id: "teach", text: "could teach it" },
  { id: "explain", text: "could explain the key ideas" },
  { id: "gist", text: "remember the gist" },
  { id: "forgot", text: "don't remember it" },
] as const;
export type Grasp = (typeof GRASP)[number]["id"];
export const isGrasp = (s: string): s is Grasp => GRASP.some((g) => g.id === s);
const knows = (g: Grasp | undefined) => g === "teach" || g === "explain" || g === "gist";

export const BOOK_STATUS = ["want", "reading", "read", "abandoned"] as const;
export type BookStatus = (typeof BOOK_STATUS)[number];

export const RESOURCE_KINDS = ["video", "channel", "article", "course", "podcast", "paper", "lab", "site"] as const;
export const RESOURCE_STATUS = ["saved", "started", "finished", "dropped"] as const;
export type ResourceStatus = (typeof RESOURCE_STATUS)[number];

/** A subject's id in a shared map: a Wikidata item, a Library of Congress subject heading, an ESCO concept, or a link. */
const SAME_AS = /^(wikidata:Q\d+|lcsh:sh\d+|esco:\S+|https?:\/\/\S+)$/;
const ISBN = /^(97[89])?\d{9}[\dX]$/;

export type GraspLine = { on: string; grasp: Grasp; book?: string; note?: string };
export type Subject = { id: string; title: string; want?: Want; for?: string; same_as?: string; requires: string[]; part_of: string[]; grasps: GraspLine[] };
export type Book = { id: string; title: string; authors: string[]; isbn?: string; openlibrary?: string; status: BookStatus; started?: string; finished?: string; subjects: string[] };
export type Resource = { id: string; kind: string; title: string; url?: string; by?: string; platform?: string; subjects: string[]; status: ResourceStatus };

export function slugOf(name: string): string {
  const s = name.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
  if (!s) throw new Error("a book, subject or resource needs a name with letters or digits");
  return s;
}

export const cleanIsbn = (s: string) => s.replace(/^isbn:?/i, "").replace(/[\s-]/g, "").toUpperCase();
const links = (v: string | undefined, dir: string) => [...(v ?? "").matchAll(new RegExp(`\\[\\[(?:${dir.replace("/", "\\/")}\\/)?([^\\]|]+)(?:\\|[^\\]]*)?\\]\\]`, "g"))].map((m) => m[1]!.trim());
const linkList = (ids: readonly string[], dir: string) => ids.map((t) => `[[${dir}/${t}]]`).join(", ");
const q = (s: string) => JSON.stringify(s);
const oneLine = (s: string) => s.replace(/\s+/g, " ").trim();
const S = "learn/subjects";

function body(text: string): { head: Record<string, string>; history: string[] } {
  const t = text.replace(/\r\n/g, "\n");
  const end = t.startsWith("---\n") ? t.indexOf("\n---\n", 4) : -1;
  const rest = end > 0 ? t.slice(end + 5) : t;
  return { head: headOf(t), history: rest.split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2)) };
}

const page = (head: string[], title: string, lines: readonly string[]) => `---\n${head.join("\n")}\n---\n\n# ${title}\n\n${lines.map((h) => `- ${h}`).join("\n")}\n`;

/** A grasp line: `2026-11-11 grasp explain [[learn/books/x]]: a note`. Other history lines are not grasps. */
export function parseGrasp(line: string): GraspLine | null {
  const m = /^(\d{4}-\d{2}(?:-\d{2})?) grasp ([a-z]+)(?: \[\[(?:learn\/books\/)?([^\]]+)\]\])?(?:: (.*))?$/.exec(line);
  if (!m || !isGrasp(m[2]!)) return null;
  return { on: m[1]!, grasp: m[2], ...(m[3] ? { book: m[3] } : {}), ...(m[4] ? { note: m[4] } : {}) };
}

const graspLine = (g: GraspLine) => `${g.on} grasp ${g.grasp}${g.book ? ` [[learn/books/${g.book}]]` : ""}${g.note ? `: ${oneLine(g.note)}` : ""}`;

export function parseSubject(id: string, text: string): Subject {
  const { head, history } = body(text);
  const want = (WANTS as readonly string[]).includes(head.want ?? "") ? (head.want as Want) : undefined;
  return {
    id,
    title: head.title ?? id,
    ...(want ? { want } : {}),
    ...(head.for ? { for: head.for } : {}),
    ...(head.same_as ? { same_as: head.same_as } : {}),
    requires: links(head.requires, S),
    part_of: links(head.part_of, S),
    grasps: history.map(parseGrasp).filter((g): g is GraspLine => g !== null),
  };
}

function subjectText(s: Subject, lines: readonly string[]): string {
  const head = [`subject: ${s.id}`, `title: ${q(s.title)}`];
  if (s.want) head.push(`want: ${s.want}`);
  if (s.for) head.push(`for: ${q(s.for)}`);
  if (s.same_as) head.push(`same_as: ${s.same_as}`);
  if (s.requires.length) head.push(`requires: ${linkList(s.requires, S)}`);
  if (s.part_of.length) head.push(`part_of: ${linkList(s.part_of, S)}`);
  return page(head, s.title, lines);
}

export function parseBook(id: string, text: string): Book {
  const { head } = body(text);
  return {
    id,
    title: head.title ?? id,
    authors: (head.authors ?? "").split(";").map((a) => a.trim()).filter(Boolean),
    ...(head.isbn ? { isbn: head.isbn } : {}),
    ...(head.openlibrary ? { openlibrary: head.openlibrary } : {}),
    status: (BOOK_STATUS as readonly string[]).includes(head.status ?? "") ? (head.status as BookStatus) : "want",
    ...(head.started ? { started: head.started } : {}),
    ...(head.finished ? { finished: head.finished } : {}),
    subjects: links(head.subjects, S),
  };
}

function bookText(b: Book, lines: readonly string[]): string {
  const head = [`book: ${b.id}`, `title: ${q(b.title)}`];
  if (b.authors.length) head.push(`authors: ${q(b.authors.join("; "))}`);
  if (b.isbn) head.push(`isbn: ${b.isbn}`);
  if (b.openlibrary) head.push(`openlibrary: ${b.openlibrary}`);
  head.push(`status: ${b.status}`);
  if (b.started) head.push(`started: ${b.started}`);
  if (b.finished) head.push(`finished: ${b.finished}`);
  if (b.subjects.length) head.push(`subjects: ${linkList(b.subjects, S)}`);
  return page(head, b.title, lines);
}

export function parseResource(id: string, text: string): Resource {
  const { head } = body(text);
  return {
    id,
    kind: head.kind ?? "",
    title: head.title ?? id,
    ...(head.url ? { url: head.url } : {}),
    ...(head.by ? { by: head.by } : {}),
    ...(head.platform ? { platform: head.platform } : {}),
    subjects: links(head.subjects, S),
    status: (RESOURCE_STATUS as readonly string[]).includes(head.status ?? "") ? (head.status as ResourceStatus) : "saved",
  };
}

function resourceText(r: Resource, lines: readonly string[]): string {
  const head = [`resource: ${r.id}`, `kind: ${r.kind}`, `title: ${q(r.title)}`];
  if (r.url) head.push(`url: ${r.url}`);
  if (r.by) head.push(`by: ${q(r.by)}`);
  if (r.platform) head.push(`platform: ${r.platform}`);
  if (r.subjects.length) head.push(`subjects: ${linkList(r.subjects, S)}`);
  head.push(`status: ${r.status}`);
  return page(head, r.title, lines);
}

const DAY = /^\d{4}-\d{2}(-\d{2})?$/;

export type SubjectChange = { title?: string; want?: Want; for?: string; same_as?: string; requires?: string[]; part_of?: string[]; grasp?: { grasp: Grasp; book?: string; note?: string; on?: string } };

/** Keep a subject: what the person wants of it, where it sits, or their grasp of it. Each change adds a dated line. */
export async function keepSubject(root: string, name: string, change: SubjectChange, now: Date): Promise<Subject> {
  const id = slugOf(name);
  if (change.want === "need" && !change.for) throw new Error("a need names what it is for: an intent, a venture, a need or a role (--for)");
  if (change.same_as && !SAME_AS.test(change.same_as)) throw new Error("--same-as is wikidata:Q<number>, lcsh:sh<number>, esco:<uri> or a link");
  if (change.grasp && !isGrasp(change.grasp.grasp)) throw new Error(`grasp is one of ${GRASP.map((g) => `${g.id} (${g.text})`).join(", ")}`);
  if (change.grasp?.on && !DAY.test(change.grasp.on)) throw new Error("--on is YYYY-MM or YYYY-MM-DD");
  if (change.grasp?.book && (await readRecord(root, join(BOOKS, `${slugOf(change.grasp.book)}.md`))) === null) throw new Error(`no book ${change.grasp.book}; keep it first: reflect learn book <title>`);
  const rel = join(SUBJECTS, `${id}.md`);
  const old = await readRecord(root, rel);
  const prev: Subject = old ? parseSubject(id, old) : { id, title: name.trim(), requires: [], part_of: [], grasps: [] };
  const s: Subject = {
    ...prev,
    ...(change.title ? { title: change.title } : {}),
    ...(change.want ? { want: change.want } : {}),
    ...(change.want === "need" ? { for: change.for } : {}),
    ...(change.same_as ? { same_as: change.same_as } : {}),
    requires: [...new Set([...prev.requires, ...(change.requires ?? []).map(slugOf)])],
    part_of: [...new Set([...prev.part_of, ...(change.part_of ?? []).map(slugOf)])],
  };
  if (change.want === "aspire") delete s.for;
  if (s.requires.includes(id) || s.part_of.includes(id)) throw new Error("a subject cannot require itself or be part of itself");
  const at = now.toISOString();
  const day = at.slice(0, 10);
  const lines = old ? body(old).history : [];
  if (change.want) lines.push(`${day} want ${change.want}${change.want === "need" ? ` for ${oneLine(change.for!)}` : ""}`);
  if (change.grasp) {
    const g: GraspLine = { on: change.grasp.on ?? day, grasp: change.grasp.grasp, ...(change.grasp.book ? { book: slugOf(change.grasp.book) } : {}), ...(change.grasp.note ? { note: change.grasp.note } : {}) };
    lines.push(graspLine(g));
    s.grasps = [...s.grasps, g];
  }
  if (!old && !lines.length) lines.push(`${day} noted`);
  await writeRecord(root, rel, subjectText(s, lines), old ? "append" : "create", at);
  return s;
}

export type BookChange = { title?: string; authors?: string[]; isbn?: string; openlibrary?: string; status?: BookStatus; subjects?: string[]; on?: string };

/** Keep a book: what it is, how far the person is with it, and the subjects it covers. */
export async function keepBook(root: string, name: string, change: BookChange, now: Date): Promise<Book> {
  const id = slugOf(name);
  if (change.status && !(BOOK_STATUS as readonly string[]).includes(change.status)) throw new Error(`--status is one of ${BOOK_STATUS.join(", ")}`);
  const isbn = change.isbn ? cleanIsbn(change.isbn) : undefined;
  if (isbn && !ISBN.test(isbn)) throw new Error("--isbn has 10 or 13 digits");
  if (change.on && !DAY.test(change.on)) throw new Error("--on is YYYY-MM or YYYY-MM-DD");
  const rel = join(BOOKS, `${id}.md`);
  const old = await readRecord(root, rel);
  const prev: Book = old ? parseBook(id, old) : { id, title: name.trim(), authors: [], status: "want", subjects: [] };
  const at = now.toISOString();
  const day = change.on ?? at.slice(0, 10);
  const b: Book = {
    ...prev,
    ...(change.title ? { title: change.title } : {}),
    ...(change.authors?.length ? { authors: change.authors } : {}),
    ...(isbn ? { isbn } : {}),
    ...(change.openlibrary ? { openlibrary: change.openlibrary } : {}),
    ...(change.status ? { status: change.status } : {}),
    subjects: [...new Set([...prev.subjects, ...(change.subjects ?? []).map(slugOf)])],
  };
  if (change.status === "reading" && !b.started) b.started = day;
  if (change.status === "read") b.finished = day;
  const lines = old ? body(old).history : [];
  if (!old || (change.status && change.status !== prev.status)) lines.push(`${day} ${b.status}`);
  await writeRecord(root, rel, bookText(b, lines), old ? "append" : "create", at);
  return b;
}

export type ResourceChange = { kind?: string; title?: string; url?: string; by?: string; platform?: string; subjects?: string[]; status?: ResourceStatus };

/** Keep a resource: a link the person saved for what they want to learn, and how far they are with it. */
export async function keepResource(root: string, name: string, change: ResourceChange, now: Date): Promise<Resource> {
  const id = slugOf(name);
  const rel = join(RESOURCES, `${id}.md`);
  const old = await readRecord(root, rel);
  if (change.kind === "book") throw new Error("a book is kept with: reflect learn book <title>");
  if (!old && !change.kind) throw new Error(`a new resource needs --kind: ${RESOURCE_KINDS.join(", ")}`);
  if (change.kind && !(RESOURCE_KINDS as readonly string[]).includes(change.kind)) throw new Error(`--kind is one of ${RESOURCE_KINDS.join(", ")}`);
  if (change.status && !(RESOURCE_STATUS as readonly string[]).includes(change.status)) throw new Error(`--status is one of ${RESOURCE_STATUS.join(", ")}`);
  if (change.url && !/^(https?:\/\/\S+|doi:\S+)$/i.test(change.url)) throw new Error("--url is a link or doi:<id>");
  if (change.platform && !(await platformIds(root)).includes(change.platform)) throw new Error(`no platform ${change.platform}; connect it first: reflect connect ${change.platform} --name <name> --kind platform`);
  const prev: Resource = old ? parseResource(id, old) : { id, kind: change.kind!, title: name.trim(), subjects: [], status: "saved" };
  const r: Resource = {
    ...prev,
    ...(change.kind ? { kind: change.kind } : {}),
    ...(change.title ? { title: change.title } : {}),
    ...(change.url ? { url: change.url } : {}),
    ...(change.by ? { by: change.by } : {}),
    ...(change.platform ? { platform: change.platform } : {}),
    ...(change.status ? { status: change.status } : {}),
    subjects: [...new Set([...prev.subjects, ...(change.subjects ?? []).map(slugOf)])],
  };
  const at = now.toISOString();
  const lines = old ? body(old).history : [];
  if (!old || (change.status && change.status !== prev.status)) lines.push(`${at.slice(0, 10)} ${r.status}`);
  await writeRecord(root, rel, resourceText(r, lines), old ? "append" : "create", at);
  return r;
}

async function notes(root: string, dir: string): Promise<[string, string][]> {
  const names = (await readdir(join(root, dir)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md").sort();
  return Promise.all(names.map(async (n) => [n.slice(0, -3), await readFile(join(root, dir, n), "utf8")] as [string, string]));
}

async function platformIds(root: string): Promise<string[]> {
  const names = (await readdir(join(root, "connections")).catch(() => [] as string[])).filter((n) => n.endsWith(".json"));
  const out: string[] = [];
  for (const n of names) {
    try {
      const c = JSON.parse(await readFile(join(root, "connections", n), "utf8")) as { id?: string; kind?: string };
      if (c.kind === "platform" && c.id) out.push(c.id);
    } catch {
      // documentProblems reports it
    }
  }
  return out;
}

export type SubjectRow = { subject: Subject; grasp?: GraspLine; books: Book[]; resources: Resource[]; blockedBy: string[] };
export type BookRow = { book: Book; unrated: string[] };
export type LearnMap = { known: SubjectRow[]; aspire: SubjectRow[]; need: SubjectRow[]; possible: SubjectRow[]; books: BookRow[] };

/** The latest grasp the person gave, by the date they gave, then the order kept. */
export function latestGrasp(grasps: readonly GraspLine[]): GraspLine | undefined {
  return grasps.map((g, i) => [g, i] as const).sort(([a, i], [b, j]) => a.on.localeCompare(b.on) || i - j).at(-1)?.[0];
}

/**
 * The person's map. Known: subjects whose latest grasp is teach, explain or gist. Aspire and need: the
 * subjects they want and do not know. Possible next: the fringe, a subject on the map they have not asked
 * for and do not know, whose prerequisites they know, that requires something or is required by something.
 * Books: every book with the subjects it covers that the person has not rated since reading it.
 */
export async function learnMap(root: string): Promise<LearnMap> {
  const books = (await notes(root, BOOKS)).map(([id, t]) => parseBook(id, t));
  const resources = (await notes(root, RESOURCES)).map(([id, t]) => parseResource(id, t));
  const subjects = new Map((await notes(root, SUBJECTS)).map(([id, t]) => [id, parseSubject(id, t)] as const));
  const named = [...[...subjects.values()].flatMap((s) => [...s.requires, ...s.part_of]), ...books.flatMap((b) => b.subjects), ...resources.flatMap((r) => r.subjects)];
  for (const id of named) if (!subjects.has(id)) subjects.set(id, { id, title: id, requires: [], part_of: [], grasps: [] });
  const rows = new Map<string, SubjectRow>();
  for (const s of subjects.values()) {
    const grasp = latestGrasp(s.grasps);
    rows.set(s.id, { subject: s, ...(grasp ? { grasp } : {}), books: books.filter((b) => b.subjects.includes(s.id)), resources: resources.filter((r) => r.subjects.includes(s.id)), blockedBy: [] });
  }
  const known = (id: string) => knows(rows.get(id)?.grasp?.grasp);
  for (const row of rows.values()) row.blockedBy = row.subject.requires.filter((r) => !known(r));
  const all = [...rows.values()];
  const open = all.filter((r) => !known(r.subject.id));
  return {
    known: all.filter((r) => known(r.subject.id)),
    aspire: open.filter((r) => r.subject.want === "aspire"),
    need: open.filter((r) => r.subject.want === "need"),
    possible: open.filter((r) => !r.subject.want && r.blockedBy.length === 0 && (r.subject.requires.length > 0 || all.some((o) => o.subject.requires.includes(r.subject.id)))),
    books: books.map((b) => ({
      book: b,
      unrated: b.status === "read" ? b.subjects.filter((s) => !(subjects.get(s)?.grasps ?? []).some((g) => g.book === b.id || (b.finished !== undefined && g.on >= b.finished))) : [],
    })),
  };
}

/** Problems in learn/ for `reflect check`. */
export async function learnProblems(root: string): Promise<string[]> {
  const out: string[] = [];
  const platforms = await platformIds(root);
  const bookIds = new Set((await notes(root, BOOKS)).map(([id]) => id));
  const subjectNotes = await notes(root, SUBJECTS);
  for (const [id, text] of subjectNotes) {
    const f = `learn/subjects/${id}.md`;
    const { head, history } = body(text);
    if (head.subject !== id) out.push(`${f}: subject must match the file name`);
    if (head.want && !(WANTS as readonly string[]).includes(head.want)) out.push(`${f}: want is aspire or need`);
    if (head.want === "need" && !head.for) out.push(`${f}: a need names what it is for`);
    if (head.same_as && !SAME_AS.test(head.same_as)) out.push(`${f}: same_as is wikidata:Q<number>, lcsh:sh<number>, esco:<uri> or a link`);
    for (const l of history) {
      if (!/^\S+ grasp /.test(l)) continue;
      const g = parseGrasp(l);
      if (!g) out.push(`${f}: grasp line not understood: ${l}`);
      else if (g.book && !bookIds.has(g.book)) out.push(`${f}: names a book that is not in learn/books: ${g.book}`);
    }
  }
  const subjects = new Map(subjectNotes.map(([id, t]) => [id, parseSubject(id, t)] as const));
  const seen = new Set<string>();
  const cycle = (id: string, path: string[]): string | null => {
    if (path.includes(id)) return [...path.slice(path.indexOf(id)), id].join(" requires ");
    if (seen.has(id)) return null;
    for (const r of subjects.get(id)?.requires ?? []) {
      const c = cycle(r, [...path, id]);
      if (c) return c;
    }
    seen.add(id);
    return null;
  };
  for (const id of subjects.keys()) {
    const c = cycle(id, []);
    if (c) {
      out.push(`learn/subjects: prerequisites go round in a circle: ${c}`);
      break;
    }
  }
  for (const [id, text] of await notes(root, BOOKS)) {
    const f = `learn/books/${id}.md`;
    const { head } = body(text);
    if (head.book !== id) out.push(`${f}: book must match the file name`);
    if (!(BOOK_STATUS as readonly string[]).includes(head.status ?? "")) out.push(`${f}: status is one of ${BOOK_STATUS.join(", ")}`);
    if (head.isbn && !ISBN.test(head.isbn)) out.push(`${f}: isbn has 10 or 13 digits`);
    if (head.openlibrary && !/^OL\d+W$/.test(head.openlibrary)) out.push(`${f}: openlibrary is a work id like OL3737036W`);
  }
  for (const [id, text] of await notes(root, RESOURCES)) {
    const f = `learn/resources/${id}.md`;
    const { head } = body(text);
    if (head.resource !== id) out.push(`${f}: resource must match the file name`);
    if (!(RESOURCE_KINDS as readonly string[]).includes(head.kind ?? "")) out.push(`${f}: kind is one of ${RESOURCE_KINDS.join(", ")}`);
    if (!(RESOURCE_STATUS as readonly string[]).includes(head.status ?? "")) out.push(`${f}: status is one of ${RESOURCE_STATUS.join(", ")}`);
    if (head.platform && !platforms.includes(head.platform)) out.push(`${f}: names a platform that is not connected: ${head.platform}`);
  }
  return out;
}

const STOP = new Set(["the", "and", "for", "with", "from", "into", "its", "their", "this", "that", "about", "of", "in", "on", "to", "a", "an"]);
const stems = (text: string) =>
  text.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").split(/[^a-z0-9]+/).filter((w) => w.length > 1 && !STOP.has(w)).map((w) => w.replace(/(ies)$/, "y").replace(/(ing|es|s)$/, "").slice(0, 6));

/**
 * Which subjects the person named in what they recalled, by matching words (no model): a subject counts
 * as named when every word of its name appears in the recall. It only marks what they named before
 * they saw the list; their grasp is still theirs to give.
 */
export function namedSubjects(recall: string, subjects: readonly { id: string; title: string }[]): Set<string> {
  const said = new Set(stems(recall));
  const out = new Set<string>();
  for (const s of subjects) {
    const words = stems(s.title.replace(/-/g, " "));
    if (words.length && words.every((w) => said.has(w))) out.add(s.id);
  }
  return out;
}

/** A recall line on a book note: `2026-11-11 recall: what the person named without looking`. */
export const parseRecall = (line: string): { on: string; text: string } | null => {
  const m = /^(\d{4}-\d{2}(?:-\d{2})?) recall: (.*)$/.exec(line);
  return m ? { on: m[1]!, text: m[2]! } : null;
};

/** Keep what the person named of a book's subjects without looking, and say which of its subjects they named. */
export async function keepRecall(root: string, book: string, recall: string, now: Date, contents: readonly string[] = []): Promise<{ book: Book; named: string[]; unnamed: string[]; contents: string[] }> {
  const id = slugOf(book);
  const rel = join(BOOKS, `${id}.md`);
  const old = await readRecord(root, rel);
  if (old === null) throw new Error(`no book ${book}; keep it first: reflect learn book <title>`);
  const said = oneLine(recall);
  if (!said) throw new Error("name what you remember, in your own words");
  const b = parseBook(id, old);
  const at = now.toISOString();
  const lines = [...body(old).history, `${at.slice(0, 10)} recall: ${said}`];
  await writeRecord(root, rel, bookText(b, lines), "append", at);
  const subjects = new Map((await notes(root, SUBJECTS)).map(([sid, t]) => [sid, parseSubject(sid, t)] as const));
  const named = namedSubjects(said, b.subjects.map((s) => ({ id: s, title: subjects.get(s)?.title ?? s.replace(/-/g, " ") })));
  return { book: b, named: b.subjects.filter((s) => named.has(s)), unnamed: b.subjects.filter((s) => !named.has(s)), contents: [...contents] };
}
