// The wiki: a compiled view of the vault, written by the person's own AI, kept apart from the record. The
// notes and the journal are the record: the person's own words, append-only, the source of truth. wiki/ is
// derived: the AI reads new entries and writes articles about what they mean (people, projects, places,
// books in the person's life, and patterns, tensions and identities across entries), in a flat encyclopedic
// voice. Every paragraph cites the notes it rests on and a quote must appear in what it cites, so an
// uncited article is refused (citation fails closed). An article about the person's inner life (a pattern,
// a tension, an identity, a philosophy) is the AI's reading until the person confirms it; a confirmation
// holds for that text only, so a rewrite makes it a reading again. The wiki is off until the person turns it
// on, and it can be rebuilt or deleted at any time without losing anything of the record.
//
// The approach follows Farza's personal wiki skill (gist farzaa/c35ac0cf): raw entries are never edited, the
// model writes articles rather than filing facts, articles are organized by theme with few quotes, an index
// with aliases and a backlink file guide an agent, and an absorb log says which entries are done. Reflect
// changes three things: the AI is the person's own (Reflect calls no model), citations are checked, and
// readings of the person are labeled until confirmed.
import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { headExtras, parseEntryFile, pointerText, POINTER_HEAD } from "./files-journal.ts";
import { recordPaths, type Layout } from "./layout.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const WIKI = "wiki";
const ABSORB = join(WIKI, "_absorb.json");
const CONFIRMED = join(WIKI, "_confirmed.json");
const INDEX = join(WIKI, "_index.md");
const BACKLINKS = join(WIKI, "_backlinks.json");
const DRAFT = join(".reflect", "wiki-draft.json");

/** Article types that are a reading of the person: labeled the AI's until the person confirms them. */
export const READING_TYPES = ["pattern", "tension", "identity", "philosophy", "life", "assessment", "metaphor"] as const;
/** Folders whose notes an article may cite: the record, never the wiki itself. */
const CITABLE = ["journal", "ideas", "learn", "concepts", "telos", "intents", "belonging", "line", "household", "community", "polis", "needs", "life", "legacy", "sources", "documents"];
const PATH = /^[a-z0-9][a-z0-9-]*(\/[a-z0-9][a-z0-9-]*)+$/;
const LINK = /\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g;
const PROMPT_BUDGET = 80_000;

export type Article = { path: string; title: string; type: string; also: string[]; related: string[]; body: string };
export type Rejected = { path: string; why: string };
export type WikiDraft = { articles: Article[]; rejected: Rejected[]; absorbed: string[] };
type Absorb = { absorbed: Record<string, string> };
type Confirmed = Record<string, { sha: string; on: string }>;

const json = async <T>(root: string, rel: string, fallback: T): Promise<T> => {
  const t = await readRecord(root, rel);
  return t === null ? fallback : (JSON.parse(t) as T);
};
const sha = (s: string) => createHash("sha256").update(s.trim()).digest("hex").slice(0, 16);
const isOn = async (root: string) => (await readRecord(root, ABSORB)) !== null;
const needOn = async (root: string) => {
  if (!(await isOn(root))) throw new Error("the wiki is off; turn it on with: reflect wiki on");
};

const README = `# Wiki

This folder is written by your own AI, from your notes and journal, when you ask for it (\`reflect wiki prompt\`, then \`reflect wiki absorb\`). It is a view, not the record: your notes and journal are the record, and nothing here is a fact about you until you confirm it.

- Every paragraph cites the notes it rests on, and a quote must appear in what it cites. Reflect refuses an article that does not.
- Articles about your inner life (patterns, tensions, identities, philosophies) say \`reading: ai\` until you confirm one with \`reflect wiki confirm <path>\`. A rewrite makes it a reading again.
- Start at \`_index.md\`. An agent answering questions about you reads the wiki, not the journal.
- You can delete this folder at any time; nothing of the record is lost.
`;

/** Turn the wiki on: its README, an empty absorb log, an empty confirmation list and index. */
export async function wikiOn(root: string, now: Date): Promise<boolean> {
  if (await isOn(root)) return false;
  const at = now.toISOString();
  await writeRecord(root, join(WIKI, "README.md"), README, "create", at);
  await writeRecord(root, ABSORB, `${JSON.stringify({ absorbed: {} }, null, 2)}\n`, "create", at);
  await writeRecord(root, CONFIRMED, "{}\n", "create", at);
  await rebuildWikiIndex(root, now);
  return true;
}

/** Every journal entry as a source: its vault path (journal/<file stem>), date and text, read live for a pointer entry. */
export async function journalSources(root: string, layout: Layout, person: string): Promise<{ id: string; date: string; text: string }[]> {
  const dir = recordPaths(layout, person).journal;
  const names = (await readdir(join(root, dir)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md").sort();
  const out: { id: string; date: string; text: string }[] = [];
  for (const n of names) {
    const raw = (await readRecord(root, join(dir, n)))!;
    const e = parseEntryFile(n, raw);
    let text = e.parts.map((p) => p.text).join("\n\n");
    const ptr = headExtras(raw).map((l) => POINTER_HEAD.exec(l)).find(Boolean);
    if (ptr) text = (await pointerText(root, ptr[1]!, ptr[2]!.trim())) ?? text;
    out.push({ id: join(dir, n.slice(0, -3)), date: e.localDate, text });
  }
  return out;
}

async function articles(root: string): Promise<{ path: string; text: string }[]> {
  const out: { path: string; text: string }[] = [];
  const walk = async (rel: string) => {
    for (const n of (await readdir(join(root, rel), { withFileTypes: true }).catch(() => [])).sort((a, b) => a.name.localeCompare(b.name))) {
      if (n.name.startsWith("_") || n.name.startsWith(".")) continue;
      const r = join(rel, n.name);
      if (n.isDirectory()) await walk(r);
      else if (n.name.endsWith(".md") && r !== join(WIKI, "README.md")) out.push({ path: r.slice(WIKI.length + 1, -3), text: (await readRecord(root, r))! });
    }
  };
  await walk(WIKI);
  return out;
}

export const WIKI_PROMPT = `You are a writer compiling my personal wiki, in Reflect, my notes vault. Below are my vault's index, the wiki so far, and journal entries of mine you have not absorbed yet. Read each entry, understand what it means, and write wiki articles that capture that understanding: the people, projects, places, books, films and ideas in my life, and the patterns, tensions, identities and philosophies that show across entries. The wiki is a map of a mind, not a diary.

How to write:
1. Process the entries in order. For each, ask what new dimension it adds to what the wiki already says. Write a full paragraph or section for a real new facet, not a sentence.
2. An article is about the thing's role in my life, not about the thing. A book article says what it meant to me and when, not a review.
3. Organize by theme, not by date. If an article reads like an event log, rewrite it whole.
4. Write like Wikipedia: flat and factual, short sentences, past or present tense, attribution over assertion ("he described it as energizing"). No em dashes, no peacock words (legendary, deeply, truly, profound), no editorial voice (interestingly, importantly), no rhetorical questions. Quotes from my entries carry the feeling; use at most two per article.
5. Named things get an article only when there is enough for three meaningful sentences; otherwise mention them where they appear. Patterns and themes across entries deserve their own articles.
6. When you change an existing article, return it whole, rewritten as one coherent piece.

Rules Reflect enforces (it refuses an article that breaks them):
- Every paragraph, list item and table row cites at least one note it rests on, as a wikilink to the note's path, for example [[journal/2026-11-10-abc]] or [[learn/books/politics]]. Cite only notes listed below. Links to other wiki articles, like [[wiki/people/sam]], are welcome but are not citations.
- A quote in double quotes of four words or more must appear word for word in a note that paragraph cites.
- No em dash characters.
- path is "<folder>/<slug>": lowercase letters, digits and dashes, folders like people, projects, places, books, patterns, tensions, identities, philosophies, eras, decisions. Folders emerge from the material.
- type is one word, such as person, project, place, book, film, idea, event, era, decision, pattern, tension, identity, philosophy.
- Leave out health details, exact money figures and other people's private information.

Reply with one JSON code block and nothing before it:

\`\`\`json
{
  "kind": "reflect.wiki",
  "version": 1,
  "absorbed": ["journal/... every entry id below you have absorbed"],
  "articles": [
    { "path": "people/sam", "title": "Sam", "type": "person", "also": ["Samuel"], "related": ["wiki/projects/corner-bakery"], "body": "Markdown body without front matter, every paragraph citing its notes." }
  ]
}
\`\`\``;

/** The prompt for the person's AI: the instructions, the vault index, the wiki so far and up to `limit` entries not yet absorbed. */
export async function wikiPrompt(root: string, layout: Layout, person: string, limit = 15): Promise<{ text: string; entries: string[]; remaining: number }> {
  await needOn(root);
  const log = await json<Absorb>(root, ABSORB, { absorbed: {} });
  const pending = (await journalSources(root, layout, person)).filter((s) => !log.absorbed[s.id]);
  const batch = pending.slice(0, limit);
  const parts = [WIKI_PROMPT, "", "## My vault's index", "", (await readRecord(root, "index.md")) ?? "(no index.md yet; `reflect index` writes it)", ""];
  const all = await articles(root);
  const size = all.reduce((n, a) => n + a.text.length, 0);
  parts.push("## The wiki so far", "");
  if (!all.length) parts.push("(empty)");
  else if (size <= PROMPT_BUDGET) for (const a of all) parts.push(`### wiki/${a.path}`, "", a.text, "");
  else parts.push((await readRecord(root, INDEX)) ?? "", "", "(The articles are too long to include. Create new articles only, and change an existing one only if I paste it.)");
  parts.push("", `## Entries to absorb (${batch.length} of ${pending.length})`, "");
  for (const s of batch) parts.push(`### ${s.id} (${s.date})`, "", s.text, "");
  if (!batch.length) parts.push("(none: every entry is absorbed)");
  return { text: parts.join("\n"), entries: batch.map((s) => s.id), remaining: pending.length - batch.length };
}

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const strs = (v: unknown) => (Array.isArray(v) ? v.map(str).filter((s): s is string => !!s) : []);
const norm = (s: string) => s.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, " ").trim();

/** The blocks of a body that must cite: paragraphs, list items and table rows; headings, rules and table separators are exempt. */
function citeBlocks(body: string): string[] {
  const out: string[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) out.push(para.join(" "));
    para = [];
  };
  for (const line of body.replace(/\r\n/g, "\n").split("\n")) {
    const t = line.trim();
    if (!t || /^#{1,6}\s/.test(t) || /^(-{3,}|\*{3,})$/.test(t) || /^\|?[\s:|-]+\|?$/.test(t)) {
      flush();
      continue;
    }
    if (/^([-*+]|\d+\.)\s/.test(t) || t.startsWith("|")) {
      flush();
      out.push(t);
    } else para.push(t);
  }
  flush();
  return out;
}

/** Check one article against the record: its path, no em dash, every block cited, every long quote found in what it cites. */
export async function checkArticle(root: string, a: Article, sourceText: (path: string) => Promise<string | null>): Promise<string | null> {
  if (!PATH.test(a.path)) return `path must be <folder>/<slug> in lowercase letters, digits and dashes: ${a.path}`;
  if (!a.title) return "no title";
  if (!a.body.trim()) return "no body";
  if (/[—]/.test(a.body + a.title)) return "uses an em dash";
  for (const block of citeBlocks(a.body)) {
    // A "See also" line of wiki links claims nothing, so it needs no citation.
    if (/^[-*+]?\s*(see( also)?:?)?[\s,.;]*(and[\s,.;]*)?$/i.test(block.replace(LINK, (m, p: string) => (p.startsWith(`${WIKI}/`) ? "" : m)).replace(/\band\b/gi, ""))) continue;
    const cited = [...block.matchAll(LINK)].map((m) => m[1]!.trim().replace(/\.md$/, "")).filter((p) => CITABLE.includes(p.split("/")[0]!));
    const texts: string[] = [];
    for (const p of cited) {
      const t = await sourceText(p);
      if (t !== null) texts.push(norm(t));
    }
    if (!texts.length) return `a passage cites no note in the record: "${block.slice(0, 80)}"`;
    for (const m of block.matchAll(/["“]([^"“”]+)["”]/g)) {
      const quote = norm(m[1]!);
      if (quote.split(" ").length >= 4 && !texts.some((t) => t.includes(quote))) return `a quote is not in the notes its passage cites: "${quote.slice(0, 80)}"`;
    }
  }
  return null;
}

/** Read the AI's reply and check each article. Nothing is written; the result is kept as a draft. */
export async function readWikiReply(root: string, layout: Layout, person: string, reply: string): Promise<WikiDraft> {
  await needOn(root);
  const fenced = /```(?:json)?\s*\n([\s\S]*?)\n```/.exec(reply);
  let j: Record<string, unknown>;
  try {
    j = JSON.parse((fenced ? fenced[1]! : reply).trim()) as Record<string, unknown>;
  } catch {
    throw new Error("the reply has no JSON the wiki can read; paste the whole code block");
  }
  if (j.kind !== "reflect.wiki" || j.version !== 1) throw new Error('the JSON is not a wiki reply (kind "reflect.wiki", version 1)');
  const entries = new Map((await journalSources(root, layout, person)).map((s) => [s.id, s.text] as const));
  const sourceText = async (p: string) => entries.get(p) ?? (p.startsWith("journal/") ? null : await readRecord(root, `${p}.md`));
  const articlesOut: Article[] = [];
  const rejected: Rejected[] = [];
  const seen = new Set<string>();
  for (const r of Array.isArray(j.articles) ? (j.articles as Record<string, unknown>[]) : []) {
    if (!r || typeof r !== "object") continue;
    const a: Article = { path: str(r.path) ?? "", title: str(r.title) ?? "", type: (str(r.type) ?? "article").toLowerCase(), also: strs(r.also), related: strs(r.related), body: typeof r.body === "string" ? r.body.trim() : "" };
    if (seen.has(a.path)) {
      rejected.push({ path: a.path, why: "the reply has this path twice" });
      continue;
    }
    seen.add(a.path);
    const why = await checkArticle(root, a, sourceText);
    if (why) rejected.push({ path: a.path || "(no path)", why });
    else articlesOut.push(a);
  }
  const absorbed = strs(j.absorbed).filter((id) => entries.has(id));
  const draft = { articles: articlesOut, rejected, absorbed };
  // The draft is not a record: it waits in .reflect/ (git-ignored) until the person keeps articles from it.
  await mkdir(join(root, ".reflect"), { recursive: true });
  await writeFile(join(root, DRAFT), `${JSON.stringify(draft, null, 2)}\n`);
  return draft;
}

export async function readWikiDraft(root: string): Promise<WikiDraft> {
  const t = await readFile(join(root, DRAFT), "utf8").catch(() => null);
  if (t === null) throw new Error("no wiki draft; run: reflect wiki absorb <your AI's reply>");
  return JSON.parse(t) as WikiDraft;
}

const isReading = (type: string) => (READING_TYPES as readonly string[]).includes(type);
const yamlList = (xs: readonly string[]) => `[${xs.map((x) => JSON.stringify(x)).join(", ")}]`;

function articleText(a: Article, confirmedOn: string | null, at: string): string {
  const sources = [...new Set([...a.body.matchAll(LINK)].map((m) => m[1]!.trim()).filter((p) => CITABLE.includes(p.split("/")[0]!)))];
  const head = [`title: ${JSON.stringify(a.title)}`, `type: ${a.type}`, `also: ${yamlList(a.also)}`, `related: ${yamlList(a.related)}`, `sources: ${yamlList(sources)}`, `written: ${at}`];
  if (isReading(a.type)) head.push(`reading: ${confirmedOn ? `confirmed ${confirmedOn}` : "ai"}`);
  return `---\n${head.join("\n")}\n---\n\n# ${a.title}\n\n${a.body}\n`;
}

/** Keep the chosen articles of the draft (numbers from 1, or all), log the absorbed entries, rebuild the index and backlinks. */
export async function keepWikiDraft(root: string, numbers: readonly number[] | "all", now: Date): Promise<string[]> {
  await needOn(root);
  const draft = await readWikiDraft(root);
  const chosen = numbers === "all" ? draft.articles : [...new Set(numbers)].map((n) => {
    const a = draft.articles[n - 1];
    if (!a) throw new Error(`no article ${n}; the draft has ${draft.articles.length}`);
    return a;
  });
  const at = now.toISOString();
  const confirmed = await json<Confirmed>(root, CONFIRMED, {});
  const kept: string[] = [];
  for (const a of chosen) {
    const rel = join(WIKI, `${a.path}.md`);
    const old = await readRecord(root, rel);
    const c = confirmed[a.path];
    await writeRecord(root, rel, articleText(a, c && c.sha === sha(a.body) ? c.on : null, at), old === null ? "create" : "append", at);
    kept.push(`wiki/${a.path}${isReading(a.type) ? (c && c.sha === sha(a.body) ? " (confirmed)" : " (reading: ai)") : ""}`);
  }
  const log = await json<Absorb>(root, ABSORB, { absorbed: {} });
  for (const id of draft.absorbed) log.absorbed[id] ??= at;
  await writeRecord(root, ABSORB, `${JSON.stringify(log, null, 2)}\n`, "append", at);
  await rebuildWikiIndex(root, now);
  return kept;
}

function parseArticle(text: string): { head: Record<string, string>; body: string } {
  const head: Record<string, string> = {};
  let body = text;
  if (text.startsWith("---\n")) {
    const end = text.indexOf("\n---\n", 4);
    if (end > 0) {
      for (const l of text.slice(4, end).split("\n")) {
        const m = /^([a-z_]+):\s*(.*)$/.exec(l);
        if (m) head[m[1]!] = m[2]!;
      }
      body = text.slice(end + 5).replace(/^\s*# .*\n+/, "");
    }
  }
  return { head, body: body.trim() };
}

/** Confirm a reading: it becomes the person's, for this text only. */
export async function confirmReading(root: string, path: string, now: Date): Promise<void> {
  await needOn(root);
  const p = path.replace(/^wiki\//, "").replace(/\.md$/, "");
  const rel = join(WIKI, `${p}.md`);
  const text = await readRecord(root, rel);
  if (text === null) throw new Error(`no article ${rel}`);
  const { head, body } = parseArticle(text);
  if (!head.reading) throw new Error(`${rel} is a ${head.type ?? "plain"} article, not a reading of you; only patterns, tensions, identities, philosophies and the like need confirming`);
  const at = now.toISOString();
  const confirmed = await json<Confirmed>(root, CONFIRMED, {});
  confirmed[p] = { sha: sha(body), on: at.slice(0, 10) };
  await writeRecord(root, CONFIRMED, `${JSON.stringify(confirmed, null, 2)}\n`, "append", at);
  await writeRecord(root, rel, text.replace(/^reading: .*$/m, `reading: confirmed ${at.slice(0, 10)}`), "append", at);
  await rebuildWikiIndex(root, now);
}

/** Rewrite _index.md (articles by folder, with aliases and whether a reading is confirmed) and _backlinks.json. */
export async function rebuildWikiIndex(root: string, now: Date): Promise<void> {
  const all = await articles(root);
  const confirmed = await json<Confirmed>(root, CONFIRMED, {});
  const back: Record<string, string[]> = {};
  const rows: { path: string; line: string }[] = [];
  for (const a of all) {
    const { head, body } = parseArticle(a.text);
    for (const m of body.matchAll(LINK)) {
      const t = m[1]!.trim().replace(/^wiki\//, "");
      if (t !== a.path && all.some((x) => x.path === t) && !(back[t] ??= []).includes(a.path)) back[t]!.push(a.path);
    }
    const also = head.also && head.also !== "[]" ? ` (also: ${(JSON.parse(head.also) as string[]).join(", ")})` : "";
    const reading = head.reading ? (confirmed[a.path]?.sha === sha(body) ? "; confirmed by me" : "; the AI's reading, not confirmed") : "";
    rows.push({ path: a.path, line: `- [[wiki/${a.path}]] ${head.title ? (JSON.parse(head.title) as string) : a.path}${also}${reading}` });
  }
  const lines = ["---", "type: wiki-index", `generated: ${now.toISOString()}`, "---", "", "# Wiki index", "", "Written by my own AI from my notes; a view, not the record. Readings of me are marked until I confirm them.", ""];
  for (const dir of [...new Set(rows.map((r) => r.path.split("/")[0]!))].sort()) {
    lines.push(`## ${dir}`, "");
    for (const r of rows.filter((x) => x.path.split("/")[0] === dir)) lines.push(`${r.line}${back[r.path]?.length ? `; linked from ${back[r.path]!.length}` : ""}`);
    lines.push("");
  }
  if (!all.length) lines.push("(no articles yet)");
  const at = now.toISOString();
  const write = async (rel: string, text: string) => writeRecord(root, rel, text, (await readRecord(root, rel)) === null ? "create" : "append", at);
  await write(INDEX, `${lines.join("\n").replace(/\n+$/, "")}\n`);
  await write(BACKLINKS, `${JSON.stringify(back, null, 2)}\n`);
}

export type WikiStatus = { on: boolean; articles: number; byType: Record<string, number>; readings: number; confirmed: number; absorbed: number; pending: number };

export async function wikiStatus(root: string, layout: Layout, person: string): Promise<WikiStatus> {
  if (!(await isOn(root))) return { on: false, articles: 0, byType: {}, readings: 0, confirmed: 0, absorbed: 0, pending: 0 };
  const all = await articles(root);
  const confirmed = await json<Confirmed>(root, CONFIRMED, {});
  const byType: Record<string, number> = {};
  let readings = 0;
  let ok = 0;
  for (const a of all) {
    const { head, body } = parseArticle(a.text);
    byType[head.type ?? "article"] = (byType[head.type ?? "article"] ?? 0) + 1;
    if (head.reading) {
      readings++;
      if (confirmed[a.path]?.sha === sha(body)) ok++;
    }
  }
  const log = await json<Absorb>(root, ABSORB, { absorbed: {} });
  const sources = await journalSources(root, layout, person);
  return { on: true, articles: all.length, byType, readings, confirmed: ok, absorbed: sources.filter((s) => log.absorbed[s.id]).length, pending: sources.filter((s) => !log.absorbed[s.id]).length };
}

/** Problems for `reflect check`: an article whose cited note left the record. */
export async function wikiProblems(root: string): Promise<string[]> {
  if (!(await isOn(root))) return [];
  const out: string[] = [];
  for (const a of await articles(root)) {
    const { body } = parseArticle(a.text);
    for (const m of body.matchAll(LINK)) {
      const p = m[1]!.trim().replace(/\.md$/, "");
      if (CITABLE.includes(p.split("/")[0]!) && (await readRecord(root, `${p}.md`)) === null) out.push(`wiki/${a.path}.md cites a note no longer in the vault: ${p}`);
    }
  }
  return out;
}
