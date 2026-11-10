// An instance: one person's records in a folder. reflect.json is Image's pointer (who, which host),
// memory.jsonl is Memory, and people/<id>/ holds the graph.
import { randomBytes } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { lastShas, nextLine, parseMemory, sha256, verifyMemory } from "./memory.ts";

export type InstanceConfig = {
  reflect: 1;
  person: string;
  host: "files";
  /** "age" seals entries and reflections; null keeps them as text (the default, in a private repository). */
  encrypt: "age" | null;
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
  if (c.reflect !== 1 || typeof c.person !== "string" || !/^p-[0-9a-f]{8}$/.test(c.person) || c.host !== "files" || !(c.encrypt === null || c.encrypt === "age")) {
    throw new Error("reflect.json is not a Reflect instance config");
  }
  return c as InstanceConfig;
}

/** Start an instance in an empty folder (or one with no reflect.json). */
export async function initInstance(root: string, name: string, now = new Date()): Promise<InstanceConfig> {
  try {
    await readFile(join(root, "reflect.json"));
    throw new Error("this folder is already a Reflect instance");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }
  const config: InstanceConfig = { reflect: 1, person: `p-${randomBytes(4).toString("hex")}`, host: "files", encrypt: null };
  const person = join("people", config.person);
  const files: [string, string][] = [
    ["reflect.json", `${JSON.stringify(config, null, 2)}\n`],
    [join(person, "person.json"), `${JSON.stringify({ id: config.person, name }, null, 2)}\n`],
  ];
  let last = null;
  const lines = [];
  for (const [rel, bytes] of files) {
    await mkdir(join(root, rel, ".."), { recursive: true });
    await writeFile(join(root, rel), bytes);
    last = nextLine(last, "create", rel, bytes, now.toISOString());
    lines.push(last);
  }
  for (const dir of ["entries", "reflections", "intents", "telos"]) {
    await mkdir(join(root, person, dir), { recursive: true });
    await writeFile(join(root, person, dir, ".gitkeep"), "");
  }
  await writeFile(join(root, "memory.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
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
 * entries or reflections. `outside` lists records written or changed without reflect (by hand, say in
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
  const person = join("people", config.person);
  for (const g of await walk(root, join(person, "grants"))) {
    const grant = JSON.parse(await readFile(join(root, g), "utf8")) as { paths?: unknown };
    const paths = Array.isArray(grant.paths) ? grant.paths : [];
    for (const p of paths) if (typeof p === "string" && /^(entries|reflections|_private)(\/|$)/.test(p)) problems.push(`${g}: a grant can never name ${p}`);
  }
  const shas = lastShas(lines);
  const outside: string[] = [];
  for (const rel of [...(await walk(root, person)).filter((r) => !r.endsWith(".draft.json")), "reflect.json"]) {
    const bytes = await readFile(join(root, rel), "utf8");
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
