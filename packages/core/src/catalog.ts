// index.md: the vault's catalog, where the person's own AI starts reading (as in a wiki built for an agent).
// It lists every note by folder with one line each, linked by wikilink, so an agent reads the index first and
// opens only the notes it needs instead of searching the whole vault. It is derived: `reflect index` rewrites
// it from the notes, and nothing in it is a fact of its own. Journal entries are listed by date only; their
// text stays in the entries.
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { listIdeas, targetText } from "./ideas.ts";
import { journalHome } from "./home.ts";
import { recordPaths, type Layout } from "./layout.ts";
import { latestGrasp, learnMap, notes, parseResource, RESOURCES, type SubjectRow } from "./learning.ts";
import { LIFETIME_FOLDERS } from "./lifetime.ts";
import { readRecord, writeRecord } from "./record-io.ts";
import { readMap, readSources } from "./sources.ts";

export const INDEX_FILE = "index.md";
const MARK = "<!-- reflect index: rewritten by `reflect index`; edit the notes, not this file -->";
const RECENT = 30;

const md = (dir: string) => readdir(dir).then((ns) => ns.filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md").sort(), () => [] as string[]);
const dirs = async (dir: string) => {
  const out: string[] = [];
  for (const n of await readdir(dir).catch(() => [] as string[])) if (!n.startsWith(".") && (await stat(join(dir, n))).isDirectory()) out.push(n);
  return out.sort();
};
const oneLine = (s: string, max = 160) => {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
};

/** The catalog's text, from the vault as it is now. */
export async function buildIndex(root: string, layout: Layout, person: string, now: Date): Promise<string> {
  const paths = recordPaths(layout, person);
  const out: string[] = ["---", "type: index", `generated: ${now.toISOString()}`, "---", "", MARK, "", "# Index", ""];
  out.push("Start here. Each line links one note; open only the ones you need. Folders and what goes in them are at the end.", "");
  const section = (title: string, lines: string[]) => {
    if (!lines.length) return;
    out.push(`## ${title}`, "", ...lines, "");
  };

  const ideas = await listIdeas(root);
  section(
    "Ideas (my own thinking)",
    ideas.map((i) => `- [[ideas/${i.id}]] ${i.title}${i.statement ? `: ${oneLine(i.statement)}` : ""}${i.from.length ? ` (from ${i.from.map(targetText).join(", ")})` : ""}`),
  );

  const map = await learnMap(root);
  const subj = (rows: readonly SubjectRow[], note: (r: SubjectRow) => string) => rows.map((r) => `- [[learn/subjects/${r.subject.id}]] ${r.subject.title}${note(r) ? `: ${note(r)}` : ""}`);
  section("Subjects I know", subj(map.known, (r) => latestGrasp(r.subject.grasps)?.grasp ?? ""));
  section("Subjects I want or need", subj([...map.aspire, ...map.need], (r) => (r.subject.want === "need" && r.subject.for ? `need, for ${r.subject.for}` : r.subject.want ?? "")));
  section("Subjects shown or suggested, not rated by me", subj([...map.shown, ...map.suggested], (r) => [r.subject.shown ? `shown ${r.subject.shown}` : "", r.subject.suggested ? `suggested ${r.subject.suggested}` : ""].filter(Boolean).join(", ")));
  section("Books", map.books.map(({ book: b }) => `- [[learn/books/${b.id}]] ${b.title}${b.authors.length ? `, ${b.authors.join(", ")}` : ""}: ${b.status}`));
  section("Films", map.films.map((f) => `- [[learn/films/${f.id}]] ${f.title}${f.year ? ` (${f.year})` : ""}: ${f.status}`));
  section("Talks", map.talks.map((t) => `- [[learn/talks/${t.id}]] ${t.title}, ${t.speaker}${t.on ? `, ${t.on}` : ""}`));
  section("Conversations", map.conversations.map((c) => `- [[learn/conversations/${c.id}]] ${c.about}${c.with.length ? `, with ${c.with.join(", ")}` : ""}${c.on ? `, ${c.on}` : ""}`));
  section("Links", (await notes(root, RESOURCES)).map(([id, t]) => parseResource(id, t)).map((r) => `- [[learn/resources/${r.id}]] ${r.title} (${r.kind}${r.source ? `, ${r.source}` : ""}): ${r.status}`));

  section("Concepts", (await md(join(root, paths.concepts))).map((n) => `- [[${join(paths.concepts, n.slice(0, -3))}]]`));
  section("Ventures (each a Naive venture)", (await dirs(join(root, paths.person, "work"))).map((d) => `- [[${join(paths.person, "work", d)}]]`));

  const home = await journalHome(root);
  if (home.soul) {
    const n = (await md(join(home.root, home.paths.journal))).length;
    section("Journal", [`- kept in my soul vault (${n} moment${n === 1 ? "" : "s"}), not listed here`]);
  }
  const entries = home.soul ? [] : (await md(join(root, paths.journal))).reverse();
  if (entries.length) {
    const lines = entries.slice(0, RECENT).map((n) => `- [[${join(paths.journal, n.slice(0, -3))}]] ${n.slice(0, 10)}`);
    if (entries.length > RECENT) lines.push(`- and ${entries.length - RECENT} earlier, in ${paths.journal}/`);
    section(`Journal (${entries.length} entries, newest first, by date only)`, lines);
  }

  const sources = await readSources(root).catch(() => []);
  const srcLines: string[] = [];
  for (const s of sources) {
    const placed = await readMap(root, s.name);
    srcLines.push(`- [[sources/${s.name}]] ${s.name}: a folder outside the vault, ${placed.length} file(s) placed`);
  }
  section("Other folders (pointed at, not moved in)", srcLines);

  if ((await readRecord(root, join("wiki", "_index.md"))) !== null) {
    section("Wiki (written by my own AI; a view, not the record)", ["- [[wiki/_index]] the wiki's own index: articles by folder, readings of me marked until I confirm them"]);
  }

  const folders: string[] = [];
  for (const f of LIFETIME_FOLDERS) {
    const exists = (await stat(join(root, paths.person, f.path)).catch(() => null))?.isDirectory();
    if (exists) folders.push(`- ${f.path}/: ${oneLine(f.guide, 220)}`);
  }
  section("Folders", folders);
  return `${out.join("\n").replace(/\n+$/, "")}\n`;
}

/** Rewrite index.md. Returns whether it changed (the generated time aside). */
export async function writeIndex(root: string, layout: Layout, person: string, now: Date): Promise<boolean> {
  const text = await buildIndex(root, layout, person, now);
  const old = await readRecord(root, INDEX_FILE);
  const body = (t: string) => t.replace(/^generated: .*$/m, "");
  if (old !== null && body(old) === body(text)) return false;
  await writeRecord(root, INDEX_FILE, text, old === null ? "create" : "append", now.toISOString());
  return true;
}
