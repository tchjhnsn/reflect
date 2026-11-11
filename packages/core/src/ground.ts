// Grounding the soul vault in the element table. A moment is bound to what it was about (a note in the life
// vault, by its path); the elements the person confirmed on that moment then show on the note, counted by
// code and never stored, so a note's element profile is always what the confirmed readings say now. A hold
// is a belief or value in the person's own words, with how they hold it (desire, will, duty, sacred), never
// inferred. A reading of the person (pattern, tension, identity) answers to the table: it names its elements
// and cites moments whose confirmed readings show every one of them, and a tension is the two elements of one
// category held against each other. Model guesses never count; only labels the person confirmed or added do.
import { readdir } from "node:fs/promises";
import { join, normalize } from "node:path";
import { journalHome, type JournalHome } from "./home.ts";
import { readRecord, writeRecord } from "./record-io.ts";
import { ROLES, type Role } from "./bindings.ts";
import { CATEGORIES, PRIMES } from "./state.ts";
import { kept, type LabelRow } from "./store.ts";
import { HOLDS, type HoldId } from "./structure.ts";

export const BINDINGS = "bindings";
export const HOLDS_DIR = "holds";
export const OF_ME = "of-me";

/** One line of a binding file: the moment was about `to` (a life-vault path), or, with drop, no longer is. */
export type BoundLine = { to: string; role?: Role; at: string; drop?: true };
export type BindingFile = { moment: string; file: string; bound: BoundLine[] };

async function soulHome(root: string): Promise<JournalHome> {
  const home = await journalHome(root);
  if (!home.soul) throw new Error("bindings, holds and readings of you live in a soul vault; start one with: reflect soul init <folder>");
  return home;
}

/** Each moment's id and file stem, from the moments folder. */
export async function momentFiles(home: JournalHome): Promise<{ id: string; stem: string }[]> {
  const out: { id: string; stem: string }[] = [];
  for (const n of (await readdir(join(home.root, home.paths.journal)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".md") || n.toLowerCase() === "readme.md") continue;
    const t = (await readRecord(home.root, join(home.paths.journal, n))) ?? "";
    const id = /^id:\s*(\S+)\s*$/m.exec(t.slice(0, t.indexOf("\n---", 4) + 1))?.[1];
    if (id) out.push({ id, stem: n.slice(0, -3) });
  }
  return out;
}

/** The elements the person confirmed or added on a moment, by its latest labeling; empty when it has none. */
export async function confirmedElements(home: JournalHome, momentId: string): Promise<string[]> {
  const t = await readRecord(home.root, join(home.paths.reflections, `${momentId}.json`));
  if (t === null) return [];
  const f = JSON.parse(t) as { labelings?: { labels: LabelRow[] }[] };
  const last = f.labelings?.at(-1);
  return last ? kept(last.labels) : [];
}

/** A life-vault path as bindings keep it: relative, no .md, no climbing out, naming a note or folder that exists. */
async function lifeTarget(home: JournalHome, to: string): Promise<string> {
  const p = normalize(to.trim().replace(/^\[\[|\]\]$/g, "")).replace(/\\/g, "/").replace(/\.md$/, "").replace(/\/+$/, "");
  if (!p || p === "." || p.startsWith("..") || p.startsWith("/") || p.startsWith(".")) throw new Error(`not a path in your life vault: ${to}`);
  const { stat } = await import("node:fs/promises");
  const hit = (await stat(join(home.lifeRoot, `${p}.md`)).catch(() => null)) ?? (await stat(join(home.lifeRoot, p)).catch(() => null));
  if (!hit) throw new Error(`no note or folder ${p} in your life vault`);
  return p;
}

const current = (f: BindingFile): { to: string; role?: Role }[] => {
  const live = new Map<string, { to: string; role?: Role }>();
  for (const l of f.bound) {
    if (l.drop) live.delete(l.to);
    else live.set(l.to, { to: l.to, ...(l.role ? { role: l.role } : {}) });
  }
  return [...live.values()];
};

async function bindingFile(home: JournalHome, momentId: string): Promise<BindingFile | null> {
  const t = await readRecord(home.root, join(BINDINGS, `${momentId}.json`));
  return t === null ? null : (JSON.parse(t) as BindingFile);
}

/** Bind a moment to what it was about. Returns false when it was already bound so. */
export async function bindMoment(root: string, momentId: string, to: string, role: string | undefined, now: Date): Promise<{ to: string; changed: boolean }> {
  const home = await soulHome(root);
  const m = (await momentFiles(home)).find((x) => x.id === momentId || x.stem === momentId);
  if (!m) throw new Error(`no moment ${momentId}; reflect list shows them`);
  if (role !== undefined && !(ROLES as readonly string[]).includes(role)) throw new Error(`a role is one of ${ROLES.join(", ")}`);
  const target = await lifeTarget(home, to);
  const old = await bindingFile(home, m.id);
  const f: BindingFile = old ?? { moment: m.id, file: join(home.paths.journal, m.stem), bound: [] };
  const now_ = current(f).find((b) => b.to === target);
  if (now_ && now_.role === role) return { to: target, changed: false };
  const at = now.toISOString();
  f.bound.push({ to: target, ...(role ? { role: role as Role } : {}), at });
  await writeRecord(home.root, join(BINDINGS, `${m.id}.json`), `${JSON.stringify(f, null, 2)}\n`, old ? "append" : "create", at);
  return { to: target, changed: true };
}

/** Unbind: the line stays in the file and memory; the binding no longer counts. */
export async function unbindMoment(root: string, momentId: string, to: string, now: Date): Promise<boolean> {
  const home = await soulHome(root);
  const m = (await momentFiles(home)).find((x) => x.id === momentId || x.stem === momentId);
  if (!m) throw new Error(`no moment ${momentId}`);
  const f = await bindingFile(home, m.id);
  const p = normalize(to.trim()).replace(/\.md$/, "").replace(/\/+$/, "");
  if (!f || !current(f).some((b) => b.to === p)) return false;
  const at = now.toISOString();
  f.bound.push({ to: p, at, drop: true });
  await writeRecord(home.root, join(BINDINGS, `${m.id}.json`), `${JSON.stringify(f, null, 2)}\n`, "append", at);
  return true;
}

/** What a moment is bound to now. */
export async function bindingsOf(root: string, momentId: string): Promise<{ to: string; role?: Role }[]> {
  const home = await soulHome(root);
  const f = await bindingFile(home, momentId);
  return f ? current(f) : [];
}

export type Profile = {
  note: string;
  /** Moments bound to the note, or to a note under it when it is a folder. */
  moments: string[];
  /** Of those, how many have no confirmed reading yet. */
  unread: number;
  /** Each confirmed element and how many of the bound moments show it, most first. */
  elements: { element: string; moments: number }[];
};

/** A life note's element profile, counted from the confirmed readings of the moments bound to it. */
export async function noteProfile(root: string, note: string): Promise<Profile> {
  const home = await soulHome(root);
  const p = normalize(note.trim()).replace(/\.md$/, "").replace(/\/+$/, "");
  const moments: string[] = [];
  const count = new Map<string, number>();
  let unread = 0;
  for (const n of (await readdir(join(home.root, BINDINGS)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".json")) continue;
    const f = JSON.parse((await readRecord(home.root, join(BINDINGS, n)))!) as BindingFile;
    if (!current(f).some((b) => b.to === p || b.to.startsWith(`${p}/`))) continue;
    moments.push(f.moment);
    const els = await confirmedElements(home, f.moment);
    if (!els.length) unread++;
    for (const e of new Set(els)) count.set(e, (count.get(e) ?? 0) + 1);
  }
  const elements = [...count].map(([element, n]) => ({ element, moments: n })).sort((a, b) => b.moments - a.moments || a.element.localeCompare(b.element));
  return { note: p, moments, unread, elements };
}

// Holds.

export const HOLD_KINDS = ["belief", "value"] as const;
export type HoldKind = (typeof HOLD_KINDS)[number];
/** How a hold can be held: the holds of the element table that face an element. */
export const HELD_AS: readonly HoldId[] = HOLDS.filter((h) => h.prime !== null).map((h) => h.id);
const HOLD_LINE = /^- (\d{4}-\d{2}-\d{2}) stated \((belief|value)(?:, held as ([a-z_]+))?\): (.+)$/;

export type Hold = { id: string; kind: HoldKind; held: HoldId | null; element: string | null; words: string; on: string; stated: number };

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").split("-").slice(0, 6).join("-").slice(0, 60);

/** Keep a belief or value in the person's words. Stating it again (same name) adds a line; nothing is rewritten. */
export async function keepHold(root: string, words: string, kind: string, held: string | undefined, name: string | undefined, now: Date): Promise<Hold> {
  const home = await soulHome(root);
  const w = words.replace(/\s+/g, " ").trim();
  if (!w) throw new Error("a hold needs your words");
  if (!(HOLD_KINDS as readonly string[]).includes(kind)) throw new Error(`--as is ${HOLD_KINDS.join(" or ")}`);
  if (held !== undefined && !(HELD_AS as readonly string[]).includes(held)) throw new Error(`--held is one of ${HELD_AS.join(", ")}`);
  const id = slug(name ?? w);
  if (!id) throw new Error("give the hold a name with --name");
  const rel = join(HOLDS_DIR, `${id}.md`);
  const old = await readRecord(home.root, rel);
  const at = now.toISOString();
  const line = `- ${at.slice(0, 10)} stated (${kind}${held ? `, held as ${held}` : ""}): ${w}`;
  const text = old === null ? `---\ntype: hold\n---\n\n# ${name ?? w.split(" ").slice(0, 8).join(" ")}\n\n${line}\n` : `${old.replace(/\n*$/, "")}\n${line}\n`;
  await writeRecord(home.root, rel, text, old === null ? "create" : "append", at);
  return (await listHolds(root)).find((h) => h.id === id)!;
}

/** Every hold, as last stated. */
export async function listHolds(root: string): Promise<Hold[]> {
  const home = await soulHome(root);
  const out: Hold[] = [];
  for (const n of (await readdir(join(home.root, HOLDS_DIR)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".md") || n.toLowerCase() === "readme.md") continue;
    const lines = ((await readRecord(home.root, join(HOLDS_DIR, n))) ?? "").split("\n").map((l) => HOLD_LINE.exec(l)).filter((m): m is RegExpExecArray => m !== null);
    const last = lines.at(-1);
    if (!last) continue;
    const held = (last[3] as HoldId | undefined) ?? null;
    out.push({ id: n.slice(0, -3), kind: last[2] as HoldKind, held, element: held ? (HOLDS.find((h) => h.id === held)?.prime ?? null) : null, words: last[4]!, on: last[1]!, stated: lines.length });
  }
  return out;
}

/** Problems with the holds folder for `reflect check`: every hold file has at least one well-formed stated line. */
export async function holdProblems(soulRoot: string): Promise<string[]> {
  const out: string[] = [];
  for (const n of (await readdir(join(soulRoot, HOLDS_DIR)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".md") || n.toLowerCase() === "readme.md") continue;
    const t = (await readRecord(soulRoot, join(HOLDS_DIR, n))) ?? "";
    const stated = t.split("\n").filter((l) => l.startsWith("- ") && / stated /.test(l));
    if (!stated.length) out.push(`${HOLDS_DIR}/${n} states nothing`);
    for (const l of stated) {
      const m = HOLD_LINE.exec(l);
      if (!m) out.push(`${HOLDS_DIR}/${n}: not a stated line: ${l.slice(0, 80)}`);
      else if (m[3] && !(HELD_AS as readonly string[]).includes(m[3])) out.push(`${HOLDS_DIR}/${n}: not a way to hold: ${m[3]}`);
    }
  }
  return out;
}

// Readings of the person against the table.

export const GROUNDED_TYPES = ["pattern", "tension", "identity"] as const;
export const isGrounded = (type: string) => (GROUNDED_TYPES as readonly string[]).includes(type);
const PAIRS: readonly (readonly string[])[] = CATEGORIES.map((c) => c.pair);

/**
 * Whether a reading answers to the element table: its elements are elements, a tension is the pair of one
 * category, it cites at least one moment, and every moment it cites has a confirmed reading showing every
 * element it names. `moments` maps a cited path (`moments/<stem>` or `journal/<stem>`) to that moment's
 * confirmed elements, or null when there is no such moment.
 */
export function groundProblem(type: string, elements: readonly string[], cited: ReadonlyMap<string, readonly string[] | null>): string | null {
  if (!isGrounded(type)) return null;
  const an = type === "identity" ? "an" : "a";
  if (!elements.length) return `${an} ${type} names its elements (elements: will, desire)`;
  const bad = elements.filter((e) => !PRIMES.includes(e));
  if (bad.length) return `not an element: ${bad.join(", ")}`;
  if (type === "tension" && !(elements.length === 2 && PAIRS.some((p) => p.includes(elements[0]!) && p.includes(elements[1]!) && elements[0] !== elements[1]))) {
    return `a tension is the two elements of one category (${PAIRS.map((p) => p.join(" and ")).join("; ")})`;
  }
  if (!cited.size) return `${an} ${type} cites the moments that show it`;
  for (const [path, els] of cited) {
    if (els === null) return `no moment ${path}`;
    const missing = elements.filter((e) => !els.includes(e));
    if (missing.length) return `${path} has no confirmed reading showing ${missing.join(", ")}`;
  }
  return null;
}

const LINK = /\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g;

/** The moments a body cites, each with its confirmed elements (null when no such moment). */
export async function citedMoments(home: JournalHome, body: string): Promise<Map<string, string[] | null>> {
  const files = await momentFiles(home);
  const out = new Map<string, string[] | null>();
  for (const m of body.matchAll(LINK)) {
    const p = m[1]!.trim().replace(/\.md$/, "");
    const [dir, stem] = [p.slice(0, p.indexOf("/")), p.slice(p.indexOf("/") + 1)];
    if (dir !== home.paths.journal || out.has(p)) continue;
    const f = files.find((x) => x.stem === stem);
    out.set(p, f ? await confirmedElements(home, f.id) : null);
  }
  return out;
}

export const elementsOf = (v: string | undefined): string[] =>
  (v ?? "").replace(/^\[|\]$/g, "").split(",").map((s) => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);

/** Problems with of-me/ for `reflect check`: each reading answers to the table. */
export async function ofMeProblems(soulRoot: string): Promise<string[]> {
  const home = await journalHome(soulRoot);
  const out: string[] = [];
  for (const n of (await readdir(join(soulRoot, OF_ME)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".md") || n.toLowerCase() === "readme.md") continue;
    const t = (await readRecord(soulRoot, join(OF_ME, n))) ?? "";
    const head: Record<string, string> = {};
    const end = t.startsWith("---\n") ? t.indexOf("\n---\n", 4) : -1;
    if (end > 0) for (const l of t.slice(4, end).split("\n")) {
      const m = /^([a-z_]+):\s*(.*)$/.exec(l);
      if (m) head[m[1]!] = m[2]!;
    }
    const type = head.type ?? "";
    if (!isGrounded(type)) {
      out.push(`${OF_ME}/${n} is not a pattern, tension or identity (type: ${type || "none"})`);
      continue;
    }
    const why = groundProblem(type, elementsOf(head.elements), await citedMoments(home, end > 0 ? t.slice(end + 5) : t));
    if (why) out.push(`${OF_ME}/${n}: ${why}`);
  }
  return out;
}

/** Problems with bindings/ for `reflect check`: each file names a moment that exists, and each live binding a life note that does. */
export async function boundProblems(soulRoot: string): Promise<string[]> {
  const home = await journalHome(soulRoot);
  const ids = new Set((await momentFiles(home)).map((m) => m.id));
  const { stat } = await import("node:fs/promises");
  const out: string[] = [];
  for (const n of (await readdir(join(soulRoot, BINDINGS)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".json")) continue;
    let f: BindingFile;
    try {
      f = JSON.parse((await readRecord(soulRoot, join(BINDINGS, n)))!) as BindingFile;
    } catch {
      out.push(`${BINDINGS}/${n} is not JSON`);
      continue;
    }
    if (`${f.moment}.json` !== n || !ids.has(f.moment)) out.push(`${BINDINGS}/${n} names no moment in this vault`);
    for (const b of current(f)) {
      const there = (await stat(join(home.lifeRoot, `${b.to}.md`)).catch(() => null)) ?? (await stat(join(home.lifeRoot, b.to)).catch(() => null));
      if (!there) out.push(`${BINDINGS}/${n} binds to ${b.to}, which is no longer in the life vault`);
    }
  }
  return out;
}

/** Every life note or folder some moment is bound to now, with how many moments. */
export async function boundNotes(root: string): Promise<{ note: string; moments: number }[]> {
  const home = await soulHome(root);
  const count = new Map<string, number>();
  for (const n of (await readdir(join(home.root, BINDINGS)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".json")) continue;
    const f = JSON.parse((await readRecord(home.root, join(BINDINGS, n)))!) as BindingFile;
    for (const b of current(f)) count.set(b.to, (count.get(b.to) ?? 0) + 1);
  }
  return [...count].map(([note, moments]) => ({ note, moments })).sort((a, b) => b.moments - a.moments || a.note.localeCompare(b.note));
}

export type OfMeReading = { path: string; title: string; type: string; elements: string[]; body: string; confirmed: string | null; problem: string | null };

/** The readings of the person in of-me/, each with whether it is confirmed and whether it still answers to the table. */
export async function ofMeReadings(root: string): Promise<OfMeReading[]> {
  const home = await soulHome(root);
  const out: OfMeReading[] = [];
  for (const n of (await readdir(join(home.root, OF_ME)).catch(() => [] as string[])).sort()) {
    if (!n.endsWith(".md") || n.toLowerCase() === "readme.md") continue;
    const t = (await readRecord(home.root, join(OF_ME, n))) ?? "";
    const head: Record<string, string> = {};
    const end = t.startsWith("---\n") ? t.indexOf("\n---\n", 4) : -1;
    if (end > 0) for (const l of t.slice(4, end).split("\n")) {
      const m = /^([a-z_]+):\s*(.*)$/.exec(l);
      if (m) head[m[1]!] = m[2]!;
    }
    const raw = (end > 0 ? t.slice(end + 5) : t).trim();
    const title = /^# (.+)$/m.exec(raw)?.[1] ?? (head.title ? (JSON.parse(head.title) as string) : n.slice(0, -3));
    const body = raw.replace(/^# .*\n+/, "").trim();
    const type = head.type ?? "";
    const elements = elementsOf(head.elements);
    const problem = isGrounded(type) ? groundProblem(type, elements, await citedMoments(home, body)) : `not a pattern, tension or identity (type: ${type || "none"})`;
    const confirmed = /^confirmed (\d{4}-\d{2}-\d{2})$/.exec(head.reading ?? "")?.[1] ?? null;
    out.push({ path: `${OF_ME}/${n.slice(0, -3)}`, title, type, elements, body, confirmed, problem });
  }
  return out;
}

/** Across all moments: how many show each element in their confirmed readings, and how many moments have any. */
export async function elementTotals(root: string): Promise<{ moments: number; read: number; elements: Record<string, number> }> {
  const home = await soulHome(root);
  const files = await momentFiles(home);
  const elements: Record<string, number> = {};
  let read = 0;
  for (const m of files) {
    const els = await confirmedElements(home, m.id);
    if (els.length) read++;
    for (const e of new Set(els)) elements[e] = (elements[e] ?? 0) + 1;
  }
  return { moments: files.length, read, elements };
}
