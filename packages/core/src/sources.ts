// Folders the person already keeps, pointed at instead of moved in. A source is a folder outside the vault,
// named in sources.json; Reflect reads it and never moves, edits or copies a file in it. Its index (in
// .reflect/, rebuilt at will) lists each file with its size, date and a sha256 fingerprint, leaving out
// hidden folders, dependencies and anything that looks like a secret. The person's own AI reads the files
// and proposes where each belongs (a book, a film, an idea, a journal entry, a venture, a document, or
// nowhere), with a reason and a confidence: a draft the person keeps item by item. A kept placement becomes a
// pointer: a dated line on the note it belongs to (`pointer notes:Reading/Politics.md #<sha>`) and a line
// in the source's map (sources/<name>.md). A journal entry is the one placement that is a whole record: an
// entry in journal/ whose front matter points at the file, and whose text the journal reads from the file
// each time, so it can be reflected on and labeled without being copied. The file stays where it is and stays the original; check says
// when a pointed-to file changed, moved or vanished.
import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { documentType, filesDocuments } from "./documents.ts";
import { keepIdea } from "./ideas.ts";
import type { Entry } from "./journal.ts";
import type { IsoDate } from "./person.ts";
import { entryFileText, headExtras, POINTER_HEAD, pointerBody } from "./files-journal.ts";
import { recordPaths, type Layout } from "./layout.ts";
import { keepBook, keepConversation, keepFilm, keepResource, keepSubject, keepTalk, slugOf } from "./learning.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const SOURCES_FILE = "sources.json";
export const MAPS = "sources";
const INDEX_DIR = join(".reflect", "index");
const DRAFT_DIR = join(".reflect", "placements");

export type Source = { name: string; path: string; exclude: string[]; added: string };
export type IndexedFile = { path: string; size: number; modified: string; sha256: string; text: boolean };

const NAME = /^[a-z0-9][a-z0-9-]{0,39}$/;
const SKIP_DIRS = new Set([".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".obsidian", ".trash", ".Trash", ".reflect", ".next", "dist", "build"]);
/** Names that may hold secrets: never indexed, never offered to an AI. */
const SECRET = /(^\.env(\..*)?$)|(\.(pem|key|p12|pfx|keystore|jks|kdbx|asc|gpg)$)|(^id_(rsa|dsa|ecdsa|ed25519))|(secret|credential|password|passwd|token|apikey|api_key)/i;
const TEXT = /\.(md|markdown|txt|text|org|rst|adoc|html?|csv|tsv|json|ya?ml|tex|rtf|opml)$/i;
const READABLE = /\.(pdf|docx?|odt|epub|pptx?|key|pages|xlsx?)$/i;

export async function readSources(root: string): Promise<Source[]> {
  const t = await readRecord(root, SOURCES_FILE);
  if (t === null) return [];
  const j = JSON.parse(t) as { sources?: Source[] };
  return j.sources ?? [];
}

const inside = (child: string, parent: string) => {
  const r = relative(parent, child);
  return r === "" || (!r.startsWith("..") && !isAbsolute(r));
};

/** Name a folder as a source. It must exist, be a folder, and not be inside the vault. */
export async function addSource(root: string, name: string, path: string, opts: { exclude?: string[] }, now: Date): Promise<Source> {
  if (!NAME.test(name)) throw new Error("a source's name is lowercase letters, digits and dashes, like notes or drive");
  const abs = resolve(path);
  const st = await stat(abs).catch(() => null);
  if (!st?.isDirectory()) throw new Error(`no folder at ${abs}`);
  if (inside(abs, resolve(root))) throw new Error("that folder is inside the vault already; a source is a folder outside it");
  const sources = await readSources(root);
  if (sources.some((s) => s.name === name)) throw new Error(`there is already a source named ${name}`);
  if (sources.some((s) => resolve(s.path) === abs)) throw new Error("that folder is already a source");
  const s: Source = { name, path: abs, exclude: (opts.exclude ?? []).map((e) => e.trim()).filter(Boolean), added: now.toISOString().slice(0, 10) };
  const old = await readRecord(root, SOURCES_FILE);
  await writeRecord(root, SOURCES_FILE, `${JSON.stringify({ sources: [...sources, s] }, null, 2)}\n`, old === null ? "create" : "append", now.toISOString());
  return s;
}

export async function getSource(root: string, name: string): Promise<Source> {
  const s = (await readSources(root)).find((x) => x.name === name);
  if (!s) throw new Error(`no source named ${name}; add one with: reflect source add <name> <folder>`);
  return s;
}

const excluded = (rel: string, patterns: readonly string[]) =>
  patterns.some((p) => {
    const q = p.replace(/^\/+|\/+$/g, "");
    if (q.startsWith("*.")) return rel.toLowerCase().endsWith(q.slice(1).toLowerCase());
    return rel === q || rel.startsWith(`${q}/`);
  });

async function sha256(file: string): Promise<string> {
  const h = createHash("sha256");
  for await (const chunk of createReadStream(file)) h.update(chunk as Buffer);
  return h.digest("hex");
}

/** Walk a source and fingerprint its files, leaving out hidden folders, dependencies, secrets, exclusions and the vault. */
export async function indexSource(root: string, name: string): Promise<{ files: IndexedFile[]; skipped: number }> {
  const s = await getSource(root, name);
  const vault = resolve(root);
  const files: IndexedFile[] = [];
  let skipped = 0;
  const walk = async (dir: string): Promise<void> => {
    const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
    for (const e of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      const abs = join(dir, e.name);
      const rel = relative(s.path, abs).split(sep).join("/");
      if (e.isSymbolicLink()) continue;
      if (e.isDirectory()) {
        if (SKIP_DIRS.has(e.name) || e.name.startsWith(".") || inside(abs, vault) || excluded(rel, s.exclude)) {
          skipped++;
          continue;
        }
        await walk(abs);
      } else if (e.isFile()) {
        if (e.name.startsWith(".") || SECRET.test(e.name) || excluded(rel, s.exclude)) {
          skipped++;
          continue;
        }
        const st = await stat(abs);
        files.push({ path: rel, size: st.size, modified: st.mtime.toISOString(), sha256: await sha256(abs), text: TEXT.test(e.name) || READABLE.test(e.name) });
      }
    }
  };
  if (!(await stat(s.path).catch(() => null))?.isDirectory()) throw new Error(`source ${name} is not reachable here: ${s.path}`);
  await walk(s.path);
  await mkdir(join(root, INDEX_DIR), { recursive: true });
  await writeFile(join(root, INDEX_DIR, `${name}.json`), `${JSON.stringify({ source: name, at: new Date().toISOString(), files }, null, 2)}\n`);
  return { files, skipped };
}

export async function readIndex(root: string, name: string): Promise<IndexedFile[] | null> {
  const t = await readFile(join(root, INDEX_DIR, `${name}.json`), "utf8").catch(() => null);
  return t ? ((JSON.parse(t) as { files: IndexedFile[] }).files ?? []) : null;
}

// --- placements -----------------------------------------------------------------------------------

export const PLACE_KINDS = ["entry", "book", "film", "talk", "link", "conversation", "subject", "idea", "venture", "document", "none"] as const;
export type PlaceKind = (typeof PLACE_KINDS)[number];

export type Placement = {
  path: string;
  kind: PlaceKind;
  /** The title, name or subject the file belongs under; for a document, its type id from `reflect documents`. */
  name?: string;
  date?: string;
  authors?: string[];
  speaker?: string;
  with?: string[];
  url?: string;
  statement?: string;
  why?: string;
  confidence?: string;
};

export const PLACEMENT_PROMPT = `You are helping me organize my own files into Reflect, my notes vault, without moving them. Below is a folder of mine and a list of its files. Read each file you can open (they are on my computer) and say where it belongs. Do not move, rename, edit or summarize any file into another place.

Where a file can belong (kind):
- entry: my own dated writing, like a journal page, in a plain text file (.md, .txt). Give its date (YYYY-MM-DD) if the file shows one.
- book: notes, highlights or a summary of a book. name is the book's title; add authors if the file names them.
- film: notes on a film, documentary or series. name is its title.
- talk: notes on a talk, lecture, sermon or video of someone speaking. name is its title; speaker is who spoke.
- link: a saved article, post or thread. name is its title; url if the file has it.
- conversation: notes of a conversation. name is what it was about; with is who it was with.
- subject: study notes on an established field (for example, statistics or urban design). name is the subject.
- idea: my own thinking: a thesis, a framing, a plan of mine. name is a short name; statement is the idea in my words, quoted from the file where you can.
- venture: material for something I am building (a business, a project, an organization). name is the venture.
- document: an official document or a scan of one (a certificate, a deed, a will, a diploma). name is its type, one of: {DOCUMENT_TYPES}.
- none: anything else, or when you cannot tell.

Rules:
1. Only what the file shows. Never invent a title, an author, a date or a link.
2. One placement per file. If a file fits more than one place, choose the one it is mostly about and say so in why.
3. why is one sentence on what in the file points there. confidence is low, medium or high.
4. Leave out health details, exact money figures and other people's private information from why.
5. If you cannot open a file, give kind none and say so.

Reply with one JSON code block and nothing before it:

\`\`\`json
{
  "kind": "reflect.placements",
  "version": 1,
  "source": "{SOURCE}",
  "placements": [
    { "path": "<the path exactly as listed>", "kind": "book", "name": "...", "authors": ["..."], "why": "...", "confidence": "high" }
  ]
}
\`\`\``;

/** The prompt for the person's own AI, with the folder and the files not yet placed. */
export async function placementPrompt(root: string, name: string, limit = 200): Promise<{ text: string; files: string[]; remaining: number }> {
  const s = await getSource(root, name);
  const index = (await readIndex(root, name)) ?? (await indexSource(root, name)).files;
  const placed = new Set((await readMap(root, name)).map((l) => l.path));
  const open = index.filter((f) => !placed.has(f.path));
  const files = open.slice(0, limit).map((f) => f.path);
  const types = (await import("./documents.ts")).DOCUMENT_TYPES.map((d) => d.id).join(", ");
  const text = `${PLACEMENT_PROMPT.replace("{SOURCE}", name).replace("{DOCUMENT_TYPES}", types)}\n\nThe folder: ${s.path}\nIts files (${files.length} of ${open.length} not yet placed):\n${files.map((f) => `- ${f}`).join("\n")}\n`;
  return { text, files, remaining: open.length - files.length };
}

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const strs = (v: unknown) => (Array.isArray(v) ? v.map(str).filter((x): x is string => !!x) : []);

/** Read the AI's reply: placements for files the source's index holds, one per file. */
export async function readPlacements(root: string, name: string, reply: string): Promise<Placement[]> {
  const fenced = /```(?:json)?\s*\n([\s\S]*?)\n```/.exec(reply);
  let j: Record<string, unknown>;
  try {
    j = JSON.parse((fenced ? fenced[1]! : reply).trim()) as Record<string, unknown>;
  } catch {
    throw new Error("the reply has no JSON to read; paste the whole code block");
  }
  if (j.kind !== "reflect.placements" || j.version !== 1) throw new Error('the JSON is not placements (kind "reflect.placements", version 1)');
  const index = new Map(((await readIndex(root, name)) ?? []).map((f) => [f.path, f] as const));
  const seen = new Set<string>();
  const out: Placement[] = [];
  for (const raw of Array.isArray(j.placements) ? (j.placements as Record<string, unknown>[]) : []) {
    const path = str(raw?.path);
    const kind = str(raw?.kind) as PlaceKind | undefined;
    if (!path || !index.has(path) || seen.has(path) || !kind || !(PLACE_KINDS as readonly string[]).includes(kind)) continue;
    seen.add(path);
    out.push({
      path,
      kind,
      ...(str(raw.name) ? { name: str(raw.name) } : {}),
      ...(str(raw.date) && /^\d{4}(-\d{2}(-\d{2})?)?$/.test(str(raw.date)!) ? { date: str(raw.date) } : {}),
      ...(strs(raw.authors).length ? { authors: strs(raw.authors) } : {}),
      ...(str(raw.speaker) ? { speaker: str(raw.speaker) } : {}),
      ...(strs(raw.with).length ? { with: strs(raw.with) } : {}),
      ...(str(raw.url) && /^https?:\/\/\S+$/.test(str(raw.url)!) ? { url: str(raw.url) } : {}),
      ...(str(raw.statement) ? { statement: str(raw.statement) } : {}),
      ...(str(raw.why) ? { why: str(raw.why) } : {}),
      ...(str(raw.confidence) ? { confidence: str(raw.confidence) } : {}),
    });
  }
  return out;
}

export async function saveDraft(root: string, name: string, placements: readonly Placement[]): Promise<void> {
  await mkdir(join(root, DRAFT_DIR), { recursive: true });
  await writeFile(join(root, DRAFT_DIR, `${name}.json`), `${JSON.stringify(placements, null, 2)}\n`);
}

export async function readDraft(root: string, name: string): Promise<Placement[]> {
  const t = await readFile(join(root, DRAFT_DIR, `${name}.json`), "utf8").catch(() => null);
  if (!t) throw new Error(`no placements drafted for ${name}; run: reflect source tag ${name} <your AI's reply>`);
  return JSON.parse(t) as Placement[];
}

export const placementLine = (p: Placement) =>
  `${p.path} -> ${p.kind}${p.name ? ` ${p.name}` : ""}${p.date ? ` (${p.date})` : ""}${p.confidence ? `; ${p.confidence}` : ""}${p.why ? `: ${p.why}` : ""}`;

// --- the map and pointers -------------------------------------------------------------------------

export type MapLine = { on: string; path: string; sha: string; kind: PlaceKind; target?: string };
const MAP_LINE = /^(\d{4}-\d{2}-\d{2}) placed "((?:[^"\\]|\\.)*)" #([0-9a-f]{12}) as ([a-z]+)(?: \[\[([^\]]+)\]\])?$/;

export function parseMapLine(line: string): MapLine | null {
  const m = MAP_LINE.exec(line);
  if (!m || !(PLACE_KINDS as readonly string[]).includes(m[4]!)) return null;
  return { on: m[1]!, path: JSON.parse(`"${m[2]!}"`) as string, sha: m[3]!, kind: m[4] as PlaceKind, ...(m[5] ? { target: m[5] } : {}) };
}

/** The source's map, latest placement per file. */
export async function readMap(root: string, name: string): Promise<MapLine[]> {
  const t = await readRecord(root, join(MAPS, `${name}.md`));
  if (t === null) return [];
  const by = new Map<string, MapLine>();
  for (const l of t.split("\n")) {
    const p = l.startsWith("- ") ? parseMapLine(l.slice(2)) : null;
    if (p) by.set(p.path, p);
  }
  return [...by.values()];
}

/** A pointer line on a note: `2026-11-11 pointer notes:Reading/Politics.md #0123456789ab`. */
export const POINTER = /^(\d{4}-\d{2}-\d{2}) pointer ([a-z0-9-]+):(.+) #([0-9a-f]{12})$/;

async function appendLine(root: string, rel: string, line: string, at: string): Promise<void> {
  const old = await readRecord(root, rel);
  if (old === null) throw new Error(`no note ${rel}`);
  await writeRecord(root, rel, `${old.replace(/\n*$/, "")}\n- ${line}\n`, "append", at);
}

/** Keep the placements the person chose (numbers from 1): each becomes a pointer on its note and a line in the source's map. */
/**
 * A journal entry that points at a file in a source: the entry's front matter names the file, and the
 * journal reads its text from there each time (files-journal.ts). The entry is closed, since the writing
 * is already done; reflecting on it and labeling it work as on any entry. A file placed again keeps the
 * same entry, with the fingerprint brought up to date.
 */
async function keepPointerEntry(root: string, journalDir: string, name: string, f: IndexedFile, date: string | undefined, at: string): Promise<string> {
  const from = `${name}:${f.path}`;
  const sha = f.sha256.slice(0, 12);
  let files: string[] = [];
  try {
    files = (await readdir(join(root, journalDir))).filter((n) => n.endsWith(".md"));
  } catch {
    // no journal yet
  }
  for (const n of files) {
    const rel = join(journalDir, n);
    const text = (await readRecord(root, rel)) ?? "";
    const extras = headExtras(text);
    const ptr = extras.map((l) => POINTER_HEAD.exec(l)).find(Boolean);
    if (!ptr || `${ptr[1]}:${ptr[2]!.trim()}` !== from) continue;
    if (!extras.includes(`pointer_sha: ${sha}`)) {
      const next = text.replace(/^pointer_sha: .*$/m, `pointer_sha: ${sha}`);
      await writeRecord(root, rel, next, "append", at);
    }
    return n.replace(/\.md$/, "");
  }
  const localDate = date && /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : f.modified.slice(0, 10);
  const id = randomUUID();
  const entry: Entry = { id, localDate: localDate as IsoDate, createdAt: at, promptId: null, closedAt: at, parts: [{ seq: 0, text: pointerBody(name, f.path), createdAt: at }] };
  const stem = `${localDate}-${id}`;
  await writeRecord(root, join(journalDir, `${stem}.md`), entryFileText(entry, [`pointer: ${from}`, `pointer_sha: ${sha}`]), "create", at);
  return stem;
}

export async function keepPlacements(root: string, name: string, numbers: readonly number[], person: string, now: Date, layout: Layout = "vault"): Promise<string[]> {
  const s = await getSource(root, name);
  const draft = await readDraft(root, name);
  const index = new Map(((await readIndex(root, name)) ?? []).map((f) => [f.path, f] as const));
  const at = now.toISOString();
  const day = at.slice(0, 10);
  const mapRel = join(MAPS, `${name}.md`);
  const kept: string[] = [];
  const lines: string[] = [];
  for (const n of [...new Set(numbers)]) {
    const p = draft[n - 1];
    if (!p) throw new Error(`no placement ${n}; the draft has ${draft.length}`);
    const f = index.get(p.path);
    if (!f) throw new Error(`${p.path} is no longer in the index; run: reflect source index ${name}`);
    const sha = f.sha256.slice(0, 12);
    const pointer = `${day} pointer ${name}:${p.path} #${sha}`;
    const from = `${name}:${p.path}`;
    const need = (what: string) => {
      if (!p.name) throw new Error(`placement ${n} (${p.path}) needs ${what}`);
      return p.name;
    };
    let target: string | undefined;
    if (p.kind === "entry") {
      if (!TEXT.test(f.path)) throw new Error(`placement ${n} (${p.path}) is not a plain text file; an entry must be text Reflect can read, like .md or .txt`);
      const journalDir = recordPaths(layout, person).journal;
      target = join(journalDir, await keepPointerEntry(root, journalDir, name, f, p.date, at));
    } else if (p.kind === "book") target = `learn/books/${(await keepBook(root, need("the book's title"), { ...(p.authors ? { authors: p.authors } : {}), source: `notes in ${from}` }, now)).id}`;
    else if (p.kind === "film") target = `learn/films/${(await keepFilm(root, need("the film's title"), { source: `notes in ${from}` }, now)).id}`;
    else if (p.kind === "talk") target = `learn/talks/${(await keepTalk(root, need("the talk's title"), { ...(p.speaker ? { speaker: p.speaker } : {}), source: `notes in ${from}` }, now)).id}`;
    else if (p.kind === "link") target = `learn/resources/${(await keepResource(root, need("the link's title"), { kind: "article", ...(p.url ? { url: p.url } : {}), source: `saved in ${from}` }, now)).id}`;
    else if (p.kind === "conversation") target = `learn/conversations/${(await keepConversation(root, { with: p.with ?? [], ...(p.date ? { on: p.date } : {}), about: need("what it was about"), source: `notes in ${from}` }, now)).id}`;
    else if (p.kind === "subject") target = `learn/subjects/${(await keepSubject(root, need("the subject"), {}, now)).id}`;
    else if (p.kind === "idea") {
      const id = slugOf(need("a short name for the idea"));
      const exists = (await readRecord(root, join("ideas", `${id}.md`))) !== null;
      if (!exists && !p.statement) throw new Error(`placement ${n} (${p.path}) is a new idea and needs its statement`);
      target = `ideas/${(await keepIdea(root, p.name!, { ...(p.statement ? { statement: p.statement } : {}), from: [{ target: { text: `my notes in ${from}` } }] }, now)).id}`;
    } else if (p.kind === "document") {
      if (!documentType(need("the document type"))) throw new Error(`placement ${n}: no document type ${p.name}; \`reflect documents\` lists them`);
      await filesDocuments(root, recordPaths(layout, person), () => now).check(p.name!, "digital-copy", { where: from });
      target = `documents/${p.name}`;
    }
    if (target && p.kind !== "document" && p.kind !== "entry") await appendLine(root, `${target}.md`, pointer, at);
    lines.push(`${day} placed ${JSON.stringify(p.path)} #${sha} as ${p.kind}${target ? ` [[${target}]]` : ""}`);
    kept.push(`${p.path} -> ${target ?? p.kind}`);
  }
  const old = await readRecord(root, mapRel);
  const head = old ?? `---\nsource: ${name}\npath: ${JSON.stringify(s.path)}\n---\n\n# ${name}\n\nWhere the files in this folder belong. The files stay where they are; each line is a placement you kept.\n\n`;
  await writeRecord(root, mapRel, `${head.replace(/\n*$/, "\n")}${old ? "" : "\n"}${lines.map((l) => `- ${l}`).join("\n")}\n`, old ? "append" : "create", at);
  return kept;
}

export type SourceStatus = { source: Source; reachable: boolean; files: number; placed: number; open: number; changed: string[]; missing: string[] };

/** For each source: whether it is reachable, how many files are placed and open, and placed files that changed or vanished since. */
export async function sourceStatus(root: string, rescan = true): Promise<SourceStatus[]> {
  const out: SourceStatus[] = [];
  for (const s of await readSources(root)) {
    const reachable = Boolean((await stat(s.path).catch(() => null))?.isDirectory());
    const index = reachable && rescan ? (await indexSource(root, s.name)).files : ((await readIndex(root, s.name)) ?? []);
    const byPath = new Map(index.map((f) => [f.path, f] as const));
    const map = await readMap(root, s.name);
    const changed: string[] = [];
    const missing: string[] = [];
    if (reachable) {
      for (const l of map) {
        const f = byPath.get(l.path);
        if (!f) missing.push(l.path);
        else if (!f.sha256.startsWith(l.sha)) changed.push(l.path);
      }
    }
    const placed = new Set(map.map((l) => l.path));
    out.push({ source: s, reachable, files: index.length, placed: map.length, open: index.filter((f) => !placed.has(f.path)).length, changed, missing });
  }
  return out;
}

/** Problems for `reflect check`: a source's map lines must read, a placed file that changed or vanished is named. */
export async function sourceProblems(root: string): Promise<string[]> {
  const out: string[] = [];
  let sources: Source[];
  try {
    sources = await readSources(root);
  } catch {
    return [`${SOURCES_FILE} is not JSON`];
  }
  for (const s of sources) {
    if (!NAME.test(s.name)) out.push(`${SOURCES_FILE}: source name not understood: ${s.name}`);
    const t = await readRecord(root, join(MAPS, `${s.name}.md`));
    if (t) for (const l of t.split("\n")) if (l.startsWith("- ") && !parseMapLine(l.slice(2))) out.push(`${MAPS}/${s.name}.md: line not understood: ${l.slice(2)}`);
  }
  for (const st of await sourceStatus(root)) {
    if (!st.reachable) continue;
    for (const p of st.changed) out.push(`${MAPS}/${st.source.name}.md: ${p} changed since it was placed; run: reflect source ${st.source.name}`);
    for (const p of st.missing) out.push(`${MAPS}/${st.source.name}.md: ${p} is no longer in the folder (moved or deleted)`);
  }
  return out;
}
