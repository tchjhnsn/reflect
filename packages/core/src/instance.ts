// An instance: one person's records in a folder. reflect.json is Image's pointer (who, which host),
// memory.jsonl is Memory, and the rest is the person's vault (layout.ts): plain markdown the person
// owns and can open in Obsidian or any editor.
import { randomBytes } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { documentProblems } from "./documents.ts";
import { recordPaths, type Layout } from "./layout.ts";
import { learnProblems } from "./learning.ts";
import { ideaProblems } from "./ideas.ts";
import { LEGACY_TEMPLATES, LIFETIME_FOLDERS, lifetimeProblems } from "./lifetime.ts";
import { readRecord, writeRecord } from "./record-io.ts";
import { planProblems, readPlan } from "./release.ts";
import { rename, rmdir } from "node:fs/promises";
import { lastShas, nextLine, parseMemory, sha256, verifyMemory } from "./memory.ts";

export type InstanceConfig = {
  reflect: 1;
  person: string;
  host: "files";
  /** "age" seals entries and reflections; null keeps them as text (the default, in a private repository). */
  encrypt: "age" | null;
  /** "vault" puts the records at the folder's root; absent means the older people/<id>/ layout. */
  layout?: "vault";
  /** Set when the instance moved to another host; writes here are then refused. */
  moved_to?: string;
};

export async function readConfig(root: string): Promise<InstanceConfig> {
  let text: string;
  try {
    text = await readFile(join(root, "reflect.json"), "utf8");
  } catch {
    throw new Error(`no reflect.json in ${root}; run \`reflect init\` first`);
  }
  const c = JSON.parse(text) as Partial<InstanceConfig>;
  if (c.reflect !== 1 || typeof c.person !== "string" || !/^p-[0-9a-f]{8}$/.test(c.person) || c.host !== "files" || !(c.encrypt === null || c.encrypt === "age") || !(c.layout === undefined || c.layout === "vault")) {
    throw new Error("reflect.json is not a Reflect instance config");
  }
  return c as InstanceConfig;
}

export const layoutOf = (c: InstanceConfig): Layout => c.layout ?? "people";

const README: Record<string, string> = {
  journal: "# Journal\n\nWhere you arrange your thoughts. One file per entry; a file you write here by hand is an entry too (name it `YYYY-MM-DD-<anything>.md`). Run `reflect spark` for a noun to write about.\n",
  telos: "# Telos\n\nThe ends you are working toward, in your own words.\n",
  documents: "# Documents\n\nA custody checklist for official documents: birth certificate, Social Security card, diplomas, deeds, will and the rest. The documents need not be here. Run `reflect documents` to see the list and check off which you have and where; a card per document lands in this folder.\n",
  intents: "# Intents\n\nWhat you mean to do, and which end each serves.\n",
};

/** Write each lifetime folder's README and the legacy templates where they are missing; each is logged in memory. */
async function ensureLifetime(root: string, now: Date): Promise<string[]> {
  const written: string[] = [];
  const files: [string, string][] = [
    ...LIFETIME_FOLDERS.map((f): [string, string] => [join(f.path, "README.md"), `# ${f.title}\n\n${f.guide}\n`]),
    ...LEGACY_TEMPLATES,
  ];
  for (const [rel, bytes] of files) {
    if ((await readRecord(root, rel)) !== null) continue;
    await writeRecord(root, rel, bytes, "create", now.toISOString());
    written.push(rel);
  }
  return written;
}

/** Where a vault made before 2026-11-11 kept things, and where they go now. */
const MOVES: [string, string][] = [
  ["ventures/household", "household"],
  ["ventures/work", "work"],
  ["ventures/community", "community"],
  ["ventures/friendship", "belonging/friends"],
  ["library", "learn"],
];

/**
 * Move a vault onto the Ventures of a Lifetime folders: the old folders' files move whole (a file
 * already at the new path is never overwritten; the old one stays and is reported), then missing
 * folders get their READMEs. Moved files are adopted in memory at their new paths.
 */
export async function upgradeVault(root: string, now = new Date()): Promise<{ moved: string[]; kept: string[]; written: string[] }> {
  const config = await readConfig(root);
  if (config.layout !== "vault") throw new Error("upgrade moves a vault; this instance uses the older people/<id> layout");
  const moved: string[] = [];
  const kept: string[] = [];
  for (const [from, to] of MOVES) {
    for (const rel of await walk(root, from)) {
      const target = join(to, rel.slice(from.length + 1));
      if ((await readRecord(root, target)) !== null) {
        kept.push(rel);
        continue;
      }
      await mkdir(join(root, target, ".."), { recursive: true });
      await rename(join(root, rel), join(root, target));
      moved.push(target);
    }
  }
  for (const dir of [...MOVES.map(([from]) => from), "ventures"]) await rmdir(join(root, dir)).catch(() => undefined);
  if (moved.length) await adopt(root, moved, now);
  const written = await ensureLifetime(root, now);
  return { moved, kept, written };
}

const GITIGNORE = ".obsidian/workspace*.json\n.obsidian/cache\n.trash/\n.DS_Store\n";

/** Start an instance in an empty folder (or one with no reflect.json). */
export async function initInstance(root: string, name: string, now = new Date()): Promise<InstanceConfig> {
  try {
    await readFile(join(root, "reflect.json"));
    throw new Error("this folder is already a Reflect instance");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }
  const config: InstanceConfig = { reflect: 1, person: `p-${randomBytes(4).toString("hex")}`, host: "files", encrypt: null, layout: "vault" };
  const p = recordPaths("vault", config.person);
  const files: [string, string][] = [
    ["reflect.json", `${JSON.stringify(config, null, 2)}\n`],
    [p.recordFiles[0]!, `${JSON.stringify({ id: config.person, name }, null, 2)}\n`],
    ...(["journal", "documents", "telos", "intents"] as const).map((d): [string, string] => [join(d, "README.md"), README[d]!]),
  ];
  let last = null;
  const lines = [];
  for (const [rel, bytes] of files) {
    await mkdir(join(root, rel, ".."), { recursive: true });
    await writeFile(join(root, rel), bytes);
    last = nextLine(last, "create", rel, bytes, now.toISOString());
    lines.push(last);
  }
  for (const dir of [p.reflections, p.concepts]) {
    await mkdir(join(root, dir), { recursive: true });
    await writeFile(join(root, dir, ".gitkeep"), "");
  }
  await writeFile(join(root, "memory.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
  await ensureLifetime(root, now);
  await writeFile(join(root, ".gitignore"), GITIGNORE);
  // The workflow that runs the reader in a GitHub repository. Not a record, so not in memory.
  await mkdir(join(root, ".github", "workflows"), { recursive: true });
  await writeFile(join(root, ".github", "workflows", "reflect.yml"), await readFile(new URL("../template/reflect.yml", import.meta.url), "utf8"));
  return config;
}

async function walk(root: string, dir: string): Promise<string[]> {
  const out: string[] = [];
  let names: import("node:fs").Dirent[];
  try {
    names = await readdir(join(root, dir), { withFileTypes: true });
  } catch {
    return out;
  }
  for (const d of names) {
    const rel = join(dir, d.name);
    if (d.isDirectory()) out.push(...(await walk(root, rel)));
    else if (d.name !== ".gitkeep") out.push(rel);
  }
  return out;
}

export type CheckResult = { problems: string[]; outside: string[] };

/**
 * Check an instance. Problems fail: a config that is not one, a broken memory chain, a grant that names
 * the journal, entries or reflections. `outside` lists records written or changed without reflect (by hand, say in
 * the GitHub app). That is allowed; `adopt` records them in memory. Drafts (`*.draft.json`) are the
 * reader's guesses, not records, so they are never in memory.
 */
export async function checkInstance(root: string): Promise<CheckResult> {
  const problems: string[] = [];
  let config: InstanceConfig;
  try {
    config = await readConfig(root);
  } catch (e) {
    return { problems: [(e as Error).message], outside: [] };
  }
  let lines: ReturnType<typeof parseMemory> = [];
  try {
    lines = parseMemory(await readFile(join(root, "memory.jsonl"), "utf8"));
  } catch {
    problems.push("memory.jsonl is missing or not JSON lines");
  }
  problems.push(...verifyMemory(lines).map((p) => `memory.jsonl ${p}`));
  const p = recordPaths(layoutOf(config), config.person);
  for (const g of await walk(root, p.grants)) {
    const grant = JSON.parse(await readFile(join(root, g), "utf8")) as { paths?: unknown };
    const paths = Array.isArray(grant.paths) ? grant.paths : [];
    for (const p of paths) if (typeof p === "string" && /^(journal|entries|reflections|_private)(\/|$)/.test(p)) problems.push(`${g}: a grant can never name ${p}`);
  }
  problems.push(...(await documentProblems(root, p)), ...(await lifetimeProblems(root, p)));
  if (!p.person) {
    try {
      problems.push(...(await planProblems(root, await readPlan(root))).problems);
    } catch {
      problems.push("legacy/release.json is not JSON");
    }
    problems.push(...(await learnProblems(root)), ...(await ideaProblems(root)));
  }
  const shas = lastShas(lines);
  const outside: string[] = [];
  const found = (await Promise.all(p.recordDirs.map((d) => walk(root, d)))).flat().filter((r) => !r.endsWith(".draft.json"));
  for (const rel of [...new Set([...found, ...p.recordFiles])]) {
    const bytes = await readFile(join(root, rel), "utf8").catch(() => null);
    if (bytes === null) continue;
    if (shas.get(rel) !== sha256(bytes)) outside.push(rel);
  }
  return { problems, outside: outside.sort() };
}

/** Record in memory the records written outside reflect, as they are now. */
export async function adopt(root: string, paths: readonly string[], now = new Date()): Promise<void> {
  const lines = parseMemory(await readFile(join(root, "memory.jsonl"), "utf8"));
  let last = lines.at(-1) ?? null;
  for (const rel of paths) {
    last = nextLine(last, "adopt", relative(root, join(root, rel)), await readFile(join(root, rel), "utf8"), now.toISOString());
    lines.push(last);
  }
  await writeFile(join(root, "memory.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
}
