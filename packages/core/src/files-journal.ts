// The folder host: one person's journal as files in an instance folder, usually a git repository
// the person owns. Every write appends a line to memory.jsonl. Nothing here is ever rewritten in
// place except by adding to it: a part is added to the end of an entry, an answer or a labeling to
// the end of a reflection file.
import { randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { makeBundle, type Bundle } from "./bundle.ts";
import type { Entry, Part, Reflection } from "./journal.ts";
import { nextLine, parseMemory, type MemoryLine } from "./memory.ts";
import type { IsoDate } from "./person.ts";
import { recordPaths, type Layout } from "./layout.ts";
import { readRecord, writeRecord } from "./record-io.ts";
import { frameById } from "./reflect.ts";
import { conceptNote, sparkHead, type SparkForm } from "./spark.ts";
import { kept, type Journal, type LabelRow, type SelfTermRow } from "./store.ts";

const PART = /^<!-- part (\d+) at (\S+) -->$/;
const DATE = /^(\d{4}-\d{2}-\d{2})/;

type Head = { id: string; local_date: string; created_at: string; prompt_id?: string; closed_at?: string };
const HEAD_KEYS = new Set(["id", "local_date", "created_at", "prompt_id", "closed_at"]);

/**
 * Front matter lines an entry carries that are not the entry's own fields: a spark's noun, or tags and
 * aliases the person added in Obsidian. Kept as they are on every later write.
 */
export function headExtras(text: string): string[] {
  const body = text.replace(/\r\n/g, "\n");
  if (!body.startsWith("---\n")) return [];
  const end = body.indexOf("\n---\n", 4);
  if (end < 0) return [];
  const out: string[] = [];
  let mine = false;
  for (const line of body.slice(4, end).split("\n")) {
    const m = /^([a-z_]+):/.exec(line);
    if (m) mine = HEAD_KEYS.has(m[1]!);
    if (!mine) out.push(line);
  }
  return out;
}

/** An entry file: optional front matter, then the parts, each after the first marked with a comment. */
export function parseEntryFile(name: string, text: string): Entry {
  const stem = name.replace(/\.md$/, "");
  let body = text.replace(/\r\n/g, "\n");
  const head: Partial<Head> = {};
  if (body.startsWith("---\n")) {
    const end = body.indexOf("\n---\n", 4);
    if (end > 0) {
      for (const line of body.slice(4, end).split("\n")) {
        const m = /^([a-z_]+):\s*(.*)$/.exec(line);
        if (m && m[2]) (head as Record<string, string>)[m[1]!] = m[2];
      }
      body = body.slice(end + 5);
    }
  }
  const localDate = (head.local_date ?? DATE.exec(stem)?.[1] ?? "1970-01-01") as IsoDate;
  const createdAt = head.created_at ?? `${localDate}T00:00:00.000Z`;
  const parts: Part[] = [];
  let seq = 0;
  let at = createdAt;
  let lines: string[] = [];
  const flush = () => {
    const t = lines.join("\n").trim();
    if (t) parts.push({ seq, text: t, createdAt: at });
  };
  for (const line of body.split("\n")) {
    const m = PART.exec(line);
    if (m) {
      flush();
      seq = Number(m[1]);
      at = m[2]!;
      lines = [];
    } else lines.push(line);
  }
  flush();
  return { id: head.id ?? stem, localDate, createdAt, promptId: head.prompt_id ?? null, closedAt: head.closed_at ?? null, parts };
}

/** The front matter line that makes an entry a pointer: its text lives in a source folder (sources.ts). */
export const POINTER_HEAD = /^pointer:\s*([a-z0-9][a-z0-9-]{0,39}):(.+)$/;
const POINTER_MAX = 512 * 1024;

/**
 * The text of a file in a named source folder, front matter left out, or null when the source or the
 * file cannot be read. The path must stay inside the source folder.
 */
export async function pointerText(root: string, source: string, path: string): Promise<string | null> {
  let sources: { name: string; path: string }[] = [];
  try {
    sources = (JSON.parse(await readFile(join(root, "sources.json"), "utf8")) as { sources?: { name: string; path: string }[] }).sources ?? [];
  } catch {
    return null;
  }
  const s = sources.find((x) => x.name === source);
  if (!s) return null;
  const file = resolve(s.path, path);
  const r = relative(resolve(s.path), file);
  if (!r || r.startsWith("..") || isAbsolute(r)) return null;
  let text: string;
  try {
    const buf = await readFile(file);
    if (buf.length > POINTER_MAX || buf.includes(0)) return null;
    text = buf.toString("utf8").replace(/\r\n/g, "\n");
  } catch {
    return null;
  }
  if (text.startsWith("---\n")) {
    const end = text.indexOf("\n---\n", 4);
    if (end > 0) text = text.slice(end + 5);
  }
  return text.trim();
}

/** The body a pointer entry keeps in the vault: what a reader of the file sees, and the text used when the source is out of reach. */
export const pointerBody = (source: string, path: string) =>
  `This entry is kept in ${source}:${path}. Reflect reads its text from there and does not copy it; that folder is not reachable right now.`;

export function entryFileText(e: Entry, extras: readonly string[] = []): string {
  const head = [`id: ${e.id}`, `local_date: ${e.localDate}`, `created_at: ${e.createdAt}`];
  if (e.promptId) head.push(`prompt_id: ${e.promptId}`);
  if (e.closedAt) head.push(`closed_at: ${e.closedAt}`);
  head.push(...extras);
  const parts = e.parts.map((p, i) => (i === 0 ? p.text : `<!-- part ${p.seq} at ${p.createdAt} -->\n\n${p.text}`));
  return `---\n${head.join("\n")}\n---\n\n${parts.join("\n\n")}\n`;
}

/** What reflecting on an entry keeps: reflection runs with their answers, and each labeling. */
export type ReflectionFile = {
  entry: string;
  reflections: Reflection[];
  labelings: { at: string; labels: LabelRow[]; terms: SelfTermRow[] }[];
};

export type FilesOptions = {
  now?: () => Date;
  id?: () => string;
  layout?: Layout;
  /** The life vault, when this journal is a soul vault's moments: its sources.json resolves pointer entries, and concept notes are kept there. */
  lifeRoot?: string;
};

/** The folder host's journal, with export and import of a whole bundle, and sparks. */
export type FilesJournal = Journal & {
  exportBundle(): Promise<Bundle>;
  importBundle(b: Bundle): Promise<void>;
  /** Write a spark: an entry on one noun, closed, naming the noun as a concept note (made the first time). Returns the entry id. */
  writeSpark(noun: string, form: SparkForm, text: string, localDate: IsoDate): Promise<string>;
  /** Write an entry that carries extra front matter (a legacy entry marked to survive the person). */
  writeWithHead(text: string, localDate: IsoDate, promptId: string | null, extras: readonly string[]): Promise<string>;
};

export function filesJournal(root: string, personId: string, opts: FilesOptions = {}): FilesJournal {
  const now = () => (opts.now ?? (() => new Date()))().toISOString();
  const newId = opts.id ?? randomUUID;
  const paths = recordPaths(opts.layout ?? "people", personId);
  const entriesDir = paths.journal;
  const reflDir = paths.reflections;

  async function read(rel: string): Promise<string | null> {
    try {
      return await readFile(join(root, rel), "utf8");
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw e;
    }
  }

  async function write(rel: string, bytes: string, op: MemoryLine["op"]): Promise<void> {
    await mkdir(dirname(join(root, rel)), { recursive: true });
    const lines = parseMemory((await read("memory.jsonl")) ?? "");
    await writeFile(join(root, rel), bytes);
    const line = nextLine(lines.at(-1) ?? null, op, rel, bytes, now());
    await writeFile(join(root, "memory.jsonl"), `${[...lines, line].map((l) => JSON.stringify(l)).join("\n")}\n`);
  }

  async function entryFiles(): Promise<{ rel: string; entry: Entry }[]> {
    let names: string[];
    try {
      names = (await readdir(join(root, entriesDir))).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md");
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw e;
    }
    return Promise.all(
      names.map(async (n) => {
        const text = (await read(join(entriesDir, n)))!;
        const entry = parseEntryFile(n, text);
        const ptr = headExtras(text).map((l) => POINTER_HEAD.exec(l)).find(Boolean);
        if (ptr) {
          const live = await pointerText(opts.lifeRoot ?? root, ptr[1]!, ptr[2]!.trim());
          if (live) entry.parts = [{ seq: 0, text: live, createdAt: entry.createdAt }, ...entry.parts.filter((p) => p.seq > 0)];
        }
        return { rel: join(entriesDir, n), entry };
      }),
    );
  }

  async function find(entryId: string): Promise<{ rel: string; entry: Entry; extras: string[] }> {
    const hit = (await entryFiles()).find((f) => f.entry.id === entryId);
    if (!hit) throw new Error("no such entry");
    // Writes use the file as it is, so a pointer entry's live text is never copied into the vault.
    const text = (await read(hit.rel)) ?? "";
    return { rel: hit.rel, entry: parseEntryFile(hit.rel.split("/").pop()!, text), extras: headExtras(text) };
  }

  async function reflFile(entryId: string): Promise<ReflectionFile> {
    const t = await read(join(reflDir, `${entryId}.json`));
    return t ? (JSON.parse(t) as ReflectionFile) : { entry: entryId, reflections: [], labelings: [] };
  }

  async function saveRefl(f: ReflectionFile, op: MemoryLine["op"]): Promise<void> {
    await write(join(reflDir, `${f.entry}.json`), `${JSON.stringify(f, null, 2)}\n`, op);
  }

  async function findReflection(reflectionId: string): Promise<{ file: ReflectionFile; r: Reflection }> {
    let names: string[] = [];
    try {
      names = (await readdir(join(root, reflDir))).filter((n) => n.endsWith(".json"));
    } catch {
      // no reflections yet
    }
    for (const n of names) {
      const file = await reflFile(n.replace(/\.json$/, ""));
      const r = file.reflections.find((x) => x.id === reflectionId);
      if (r) return { file, r };
    }
    throw new Error("no such question");
  }

  return {
    async exportBundle() {
      const entries = (await entryFiles()).map((f) => f.entry);
      const labelings = [];
      const out: Entry[] = [];
      for (const e of entries) {
        const f = await reflFile(e.id);
        out.push({ ...e, reflections: f.reflections });
        labelings.push(...f.labelings.map((l) => ({ entry: e.id, ...l })));
      }
      return makeBundle(out, labelings, now());
    },
    async importBundle(b) {
      if ((await entryFiles()).length) throw new Error("this journal already has entries; import goes only into an empty one");
      for (const e of b.entries) {
        const { reflections, labels: _labels, ...entry } = e;
        await write(join(entriesDir, `${e.id.startsWith(e.localDate) ? e.id : `${e.localDate}-${e.id}`}.md`), entryFileText(entry), "create");
        const labelings = b.labelings.filter((l) => l.entry === e.id).map(({ entry: _e, ...l }) => l);
        if (reflections?.length || labelings.length) await saveRefl({ entry: e.id, reflections: [...(reflections ?? [])], labelings }, "create");
      }
    },
    async writeSpark(noun, form, text, localDate) {
      const at = now();
      const note = join(paths.concepts, `${noun}.md`);
      if (opts.lifeRoot && opts.lifeRoot !== root) {
        // A concept is a thing in shared reality: in a soul vault's journal, its note goes to the life vault.
        if ((await readRecord(opts.lifeRoot, note)) === null) await writeRecord(opts.lifeRoot, note, conceptNote(noun, at), "create", at);
      } else if ((await read(note)) === null) await write(note, conceptNote(noun, at), "create");
      const id = newId();
      const e: Entry = { id, localDate, createdAt: at, promptId: `spark/${form}`, closedAt: at, parts: [{ seq: 0, text, createdAt: at }] };
      await write(join(entriesDir, `${localDate}-${id}.md`), entryFileText(e, sparkHead(noun, form)), "create");
      return id;
    },
    async writeWithHead(text, localDate, promptId, extras) {
      const id = newId();
      const at = now();
      const e: Entry = { id, localDate, createdAt: at, promptId, closedAt: null, parts: [{ seq: 0, text, createdAt: at }] };
      await write(join(entriesDir, `${localDate}-${id}.md`), entryFileText(e, extras), "create");
      return id;
    },
    async createEntry(text, localDate, promptId) {
      const id = newId();
      const at = now();
      const e: Entry = { id, localDate, createdAt: at, promptId, closedAt: null, parts: [{ seq: 0, text, createdAt: at }] };
      await write(join(entriesDir, `${localDate}-${id}.md`), entryFileText(e), "create");
      return id;
    },
    async addPart(entryId, text) {
      const { rel, entry, extras } = await find(entryId);
      if (entry.closedAt) throw new Error("entry is closed");
      const seq = entry.parts.length ? entry.parts.at(-1)!.seq + 1 : 0;
      await write(rel, entryFileText({ ...entry, parts: [...entry.parts, { seq, text, createdAt: now() }] }, extras), "append");
    },
    async closeEntry(entryId) {
      const { rel, entry, extras } = await find(entryId);
      if (entry.closedAt) return;
      await write(rel, entryFileText({ ...entry, closedAt: now() }, extras), "append");
    },
    async listEntries(limit = 200) {
      const all = (await entryFiles()).map((f) => f.entry).sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, limit);
      return Promise.all(
        all.map(async (e) => {
          const f = await reflFile(e.id);
          const last = f.labelings.at(-1);
          return { ...e, reflections: f.reflections, labels: last ? kept(last.labels) : [] };
        }),
      );
    },
    async entryText(entryId) {
      const hit = (await entryFiles()).find((f) => f.entry.id === entryId);
      return hit ? hit.entry.parts.map((p) => p.text).join("\n\n") : null;
    },
    async startReflection(entryId, frame) {
      if (!frameById(frame)) throw new Error("no such frame");
      await find(entryId);
      const f = await reflFile(entryId);
      const id = newId();
      await saveRefl({ ...f, reflections: [...f.reflections, { id, frame, createdAt: now(), answers: [] }] }, f.reflections.length || f.labelings.length ? "append" : "create");
      return id;
    },
    async keepAnswer(reflectionId, seq, text) {
      const { file, r } = await findReflection(reflectionId);
      const q = frameById(r.frame)?.questions[seq];
      if (!q) throw new Error("no such question");
      if (r.answers.some((a) => a.seq === seq)) throw new Error("already answered");
      const answer = { seq, facet: q.facet, stance: q.stance, question: q.text, text };
      const reflections = file.reflections.map((x) => (x.id === reflectionId ? { ...x, answers: [...x.answers, answer] } : x));
      await saveRefl({ ...file, reflections }, "append");
    },
    async keepLabels(entryId, labels, terms) {
      await find(entryId);
      const f = await reflFile(entryId);
      await saveRefl({ ...f, labelings: [...f.labelings, { at: now(), labels, terms }] }, f.reflections.length || f.labelings.length ? "append" : "create");
    },
  };
}
