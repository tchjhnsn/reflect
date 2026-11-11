// The person's own ideas, the outputs of their thinking, kept apart from what they take in (learn/). An idea
// note in ideas/ says the idea in the person's words, dated each time they restate it, and links its
// inputs: `from` an input the person says fed it, `maybe from` one they suspect did without being sure
// (influence is often not conscious). Recommendations are a separate list: things that may bear on the
// idea, each placed on a spectrum from supports to opposes, with who suggested it. A recommendation is
// something to study, never an input, until the person engages with it and says it fed the idea.
//
// Every change is a dated line, never rewritten; the head is rebuilt from the lines so links show in
// Obsidian's graph. An input or recommendation is a vault note (`[[learn/books/politics]]`) or, when there
// is no note for it, the person's words in quotes with an optional link (`"a sermon on Micah 6" <https://…>`).
import { readdir, readFile } from "node:fs/promises";
import { join, normalize } from "node:path";
import { headOf } from "./lifetime.ts";
import { BOOKS, CONVERSATIONS, FILMS, RESOURCES, slugOf, SUBJECTS, TALKS } from "./learning.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const IDEAS = "ideas";
/** A pointer to a file in one of the person's own folders (sources.ts); kept on the note, never folded. */
const POINTER_LINE = /^\d{4}-\d{2}-\d{2} pointer [a-z0-9-]+:.+ #[0-9a-f]{12}$/;

/** Where a recommendation sits against the idea, from most aligned to most adversarial. */
export const STANCES = [
  { id: "supports", text: "agrees with it and gives reasons for it" },
  { id: "extends", text: "takes it further or applies it elsewhere" },
  { id: "parallels", text: "reaches something like it by another road" },
  { id: "challenges", text: "raises problems it has to answer" },
  { id: "opposes", text: "argues the contrary" },
] as const;
export type Stance = (typeof STANCES)[number]["id"];
export const isStance = (s: string): s is Stance => STANCES.some((x) => x.id === s);

/** A vault note (`ref`, a path without .md) or the person's words for something with no note (`text`), with an optional link. */
export type Target = { ref?: string; text?: string; url?: string };
export type Input = Target & { on: string; how?: string };
export type Recommendation = Target & { on: string; stance: Stance; by: string; note?: string };
export type Idea = { id: string; title: string; statement?: string; stated: { on: string; text: string }[]; from: Input[]; maybe: Input[]; related: Recommendation[] };

const q = (s: string) => JSON.stringify(s);
const oneLine = (s: string) => s.replace(/\s+/g, " ").trim();
const DAY = "(\\d{4}-\\d{2}(?:-\\d{2})?)";
const TARGET = '(\\[\\[[^\\]|]+\\]\\]|"(?:[^"\\\\]|\\\\.)*")(?: <(\\S+)>)?';

export const targetText = (t: Target) => `${t.ref ? `[[${t.ref}]]` : q(t.text ?? "")}${t.url ? ` <${t.url}>` : ""}`;
const targetKey = (t: Target) => (t.ref ? `ref:${t.ref}` : `text:${(t.text ?? "").toLowerCase()}`);

function readTarget(raw: string, url: string | undefined): Target {
  const t: Target = raw.startsWith("[[") ? { ref: raw.slice(2, -2).trim() } : { text: JSON.parse(raw) as string };
  return url ? { ...t, url } : t;
}

export type IdeaLine =
  | { kind: "stated"; on: string; text: string }
  | { kind: "from" | "maybe"; input: Input }
  | { kind: "related"; rec: Recommendation }
  | { kind: "dropped"; on: string; target: Target };

/** One history line of an idea note, or null when it is not one of the kinds above. */
export function parseIdeaLine(line: string): IdeaLine | null {
  let m = new RegExp(`^${DAY} stated: (.+)$`).exec(line);
  if (m) return { kind: "stated", on: m[1]!, text: m[2]! };
  m = new RegExp(`^${DAY} (from|maybe from) ${TARGET}(?:: (.*))?$`).exec(line);
  if (m) return { kind: m[2] === "from" ? "from" : "maybe", input: { on: m[1]!, ...readTarget(m[3]!, m[4]), ...(m[5] ? { how: m[5] } : {}) } };
  m = new RegExp(`^${DAY} related ([a-z]+) by ([^:\\[\\]"]+?) ${TARGET}(?:: (.*))?$`).exec(line);
  if (m && isStance(m[2]!)) return { kind: "related", rec: { on: m[1]!, stance: m[2], by: m[3]!.trim(), ...readTarget(m[4]!, m[5]), ...(m[6] ? { note: m[6] } : {}) } };
  m = new RegExp(`^${DAY} dropped ${TARGET}$`).exec(line);
  if (m) return { kind: "dropped", on: m[1]!, target: readTarget(m[2]!, m[3]) };
  return null;
}

export function lineText(l: IdeaLine): string {
  if (l.kind === "stated") return `${l.on} stated: ${oneLine(l.text)}`;
  if (l.kind === "from" || l.kind === "maybe") return `${l.input.on} ${l.kind === "from" ? "from" : "maybe from"} ${targetText(l.input)}${l.input.how ? `: ${oneLine(l.input.how)}` : ""}`;
  if (l.kind === "related") return `${l.rec.on} related ${l.rec.stance} by ${oneLine(l.rec.by)} ${targetText(l.rec)}${l.rec.note ? `: ${oneLine(l.rec.note)}` : ""}`;
  if (l.kind === "dropped") return `${l.on} dropped ${targetText(l.target)}`;
  return "";
}

function split(text: string): { head: Record<string, string>; title: string; history: string[] } {
  const t = text.replace(/\r\n/g, "\n");
  const end = t.startsWith("---\n") ? t.indexOf("\n---\n", 4) : -1;
  const rest = end > 0 ? t.slice(end + 5) : t;
  return { head: headOf(t), title: /^# (.+)$/m.exec(rest)?.[1]?.trim() ?? "", history: rest.split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2)) };
}

/**
 * Fold the lines in order. A target's latest line decides where it stands: a `from` after a `maybe from`
 * moves it to from, and `dropped` takes it out of inputs and recommendations alike.
 */
export function foldIdea(id: string, title: string, lines: readonly IdeaLine[]): Idea {
  const stated: Idea["stated"] = [];
  const inputs = new Map<string, { list: "from" | "maybe"; input: Input }>();
  const related = new Map<string, Recommendation>();
  for (const l of lines) {
    if (l.kind === "stated") stated.push({ on: l.on, text: l.text });
    else if (l.kind === "from" || l.kind === "maybe") {
      const k = targetKey(l.input);
      inputs.delete(k);
      inputs.set(k, { list: l.kind, input: l.input });
      related.delete(k);
    } else if (l.kind === "related") {
      const k = targetKey(l.rec);
      if (!inputs.has(k)) related.set(k, l.rec);
    } else if (l.kind === "dropped") {
      inputs.delete(targetKey(l.target));
      related.delete(targetKey(l.target));
    }
  }
  const all = [...inputs.values()];
  return {
    id,
    title,
    ...(stated.length ? { statement: stated.at(-1)!.text } : {}),
    stated,
    from: all.filter((x) => x.list === "from").map((x) => x.input),
    maybe: all.filter((x) => x.list === "maybe").map((x) => x.input),
    related: [...related.values()],
  };
}

export function parseIdea(id: string, text: string): Idea {
  const { head, title, history } = split(text);
  return foldIdea(id, head.title ?? (title || id), history.map(parseIdeaLine).filter((l): l is IdeaLine => l !== null));
}

function ideaText(i: Idea, history: readonly string[]): string {
  const head = [`idea: ${i.id}`, `title: ${q(i.title)}`];
  const refs = (xs: readonly Target[]) => xs.filter((x) => x.ref).map((x) => `[[${x.ref}]]`).join(", ");
  if (i.from.some((x) => x.ref)) head.push(`from: ${refs(i.from)}`);
  if (i.maybe.some((x) => x.ref)) head.push(`maybe_from: ${refs(i.maybe)}`);
  if (i.related.some((x) => x.ref)) head.push(`related: ${refs(i.related)}`);
  return `---\n${head.join("\n")}\n---\n\n# ${i.title}\n\n${i.statement ?? ""}\n\n${history.map((h) => `- ${h}`).join("\n")}\n`;
}

/** Folders an input can come from, searched in this order when the person names one by its title. */
const INPUT_DIRS = [BOOKS, FILMS, TALKS, RESOURCES, CONVERSATIONS, SUBJECTS, IDEAS, "journal", "concepts"];

/**
 * What the person named: a vault path (`learn/books/politics`), a title kept in one of the input folders
 * (`Politics`), or else their own words, kept as text. A link (`https://…`) is text with that link.
 */
export async function resolveTarget(root: string, name: string): Promise<Target> {
  const n = name.trim();
  if (!n) throw new Error("name an input: a vault note, a title you kept, or your own words");
  if (/^https?:\/\/\S+$/.test(n)) return { text: n, url: n };
  const path = n.replace(/^\[\[|\]\]$/g, "").replace(/\.md$/, "");
  if (path.includes("/")) {
    const clean = normalize(path);
    if (clean.startsWith("..") || clean.startsWith("/")) throw new Error(`not a path in the vault: ${n}`);
    if ((await readRecord(root, `${clean}.md`)) !== null) return { ref: clean };
  }
  let slug: string | null = null;
  try {
    slug = slugOf(n);
  } catch {
    slug = null;
  }
  if (slug) for (const dir of INPUT_DIRS) if ((await readRecord(root, join(dir, `${slug}.md`))) !== null) return { ref: join(dir, slug) };
  return { text: n };
}

export type IdeaChange = {
  statement?: string;
  from?: { target: Target; how?: string }[];
  maybe?: { target: Target; how?: string }[];
  related?: { target: Target; stance: Stance; by: string; note?: string }[];
  drop?: Target[];
  on?: string;
};

/** Keep an idea: state or restate it, link its inputs, add recommendations, or drop any of them. Each change is a dated line. */
export async function keepIdea(root: string, name: string, change: IdeaChange, now: Date): Promise<Idea> {
  const id = slugOf(name);
  const rel = join(IDEAS, `${id}.md`);
  const old = await readRecord(root, rel);
  if (!old && !change.statement?.trim()) throw new Error("a new idea needs its statement, in your own words");
  const at = now.toISOString();
  const day = change.on ?? at.slice(0, 10);
  if (!new RegExp(`^${DAY}$`).test(day)) throw new Error("--on is YYYY-MM or YYYY-MM-DD");
  const check = async (t: Target) => {
    if (t.ref && (await readRecord(root, `${t.ref}.md`)) === null) throw new Error(`no note ${t.ref}.md in the vault; keep it first or name it in your own words`);
    if (t.url && !/^https?:\/\/\S+$/.test(t.url)) throw new Error(`not a link: ${t.url}`);
    if (!t.ref && !t.text?.trim()) throw new Error("an input or recommendation needs a note or your words");
    if (t.ref === rel.slice(0, -3)) throw new Error("an idea is not its own input");
  };
  const add: IdeaLine[] = [];
  if (change.statement?.trim()) add.push({ kind: "stated", on: day, text: change.statement.trim() });
  for (const [kind, xs] of [["from", change.from], ["maybe", change.maybe]] as const) {
    for (const x of xs ?? []) {
      await check(x.target);
      add.push({ kind, input: { on: day, ...x.target, ...(x.how?.trim() ? { how: x.how.trim() } : {}) } });
    }
  }
  for (const r of change.related ?? []) {
    if (!isStance(r.stance)) throw new Error(`a recommendation's stance is one of ${STANCES.map((s) => s.id).join(", ")}`);
    if (!r.by.trim() || /[:[\]"]/.test(r.by)) throw new Error("say who suggested it, in a few plain words (--by)");
    await check(r.target);
    add.push({ kind: "related", rec: { on: day, stance: r.stance, by: r.by.trim(), ...r.target, ...(r.note?.trim() ? { note: r.note.trim() } : {}) } });
  }
  for (const t of change.drop ?? []) add.push({ kind: "dropped", on: day, target: t });
  if (old && !add.length) throw new Error("nothing to change: restate it, or add --from, --maybe, --related or --drop");
  const prev = old ? split(old) : { head: {} as Record<string, string>, title: name.trim(), history: [] as string[] };
  const history = [...prev.history, ...add.map(lineText)];
  const idea = foldIdea(id, prev.head.title ?? (prev.title || name.trim()), history.map(parseIdeaLine).filter((l): l is IdeaLine => l !== null));
  await writeRecord(root, rel, ideaText(idea, history), old ? "append" : "create", at);
  return idea;
}

export async function listIdeas(root: string): Promise<Idea[]> {
  const names = (await readdir(join(root, IDEAS)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md").sort();
  return Promise.all(names.map(async (n) => parseIdea(n.slice(0, -3), await readFile(join(root, IDEAS, n), "utf8"))));
}

/** Problems in ideas/ for `reflect check`. */
export async function ideaProblems(root: string): Promise<string[]> {
  const out: string[] = [];
  const names = (await readdir(join(root, IDEAS)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md");
  for (const n of names) {
    const id = n.slice(0, -3);
    const f = `${IDEAS}/${n}`;
    const { head, history } = split(await readFile(join(root, IDEAS, n), "utf8"));
    if (head.idea !== id) out.push(`${f}: idea must match the file name`);
    const lines = history.map((l) => [l, parseIdeaLine(l)] as const);
    for (const [l, p] of lines) if (!p && !POINTER_LINE.test(l)) out.push(`${f}: line not understood: ${l}`);
    if (!lines.some(([, p]) => p?.kind === "stated")) out.push(`${f}: an idea is stated in the person's words (a "stated:" line)`);
    for (const [, p] of lines) {
      const t = p?.kind === "from" || p?.kind === "maybe" ? p.input : p?.kind === "related" ? p.rec : null;
      if (t?.ref && (await readRecord(root, `${t.ref}.md`)) === null) out.push(`${f}: names a note that is not in the vault: ${t.ref}`);
    }
  }
  return out;
}
