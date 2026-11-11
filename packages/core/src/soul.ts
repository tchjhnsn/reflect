// The soul vault: the inner life, kept apart from the life vault and in the terms of the Number Theory of
// the Soul. The life vault holds what the person does, knows and keeps (learn, ideas, work, household,
// documents); the soul vault holds what the person undergoes and is: moments (the journal), readings (the
// elements the person confirmed on each moment), bindings (what each moment was about, by its path in the
// life vault), holds (beliefs and values the person stated) and of-me (readings of the person that answer
// to the element table). Image (1) is soul.json, naming the person; Memory (0) is the append-only trace,
// memory.jsonl. The two vaults link one way: the soul vault names life notes, and the life vault never
// reads into the soul vault, so the soul vault can be its own private repository that no grant reaches.
import { mkdir, readdir, readFile, rename, rmdir, stat, writeFile } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";
import { adopt, readConfig, type InstanceConfig } from "./instance.ts";
import { recordPaths, SOUL_DIRS } from "./layout.ts";
import { nextLine } from "./memory.ts";
import { readRecord, writeRecord } from "./record-io.ts";

const README: Record<(typeof SOUL_DIRS)[number], string> = {
  moments: "# Moments\n\nYour journal: one file per moment, in your words. Memory keeps every one; nothing here is rewritten, only added to.\n",
  readings: "# Readings\n\nWhat you reflected on each moment and the elements you confirmed in it (arrangement and amplitude per category). A model may propose; only what you confirm is kept as yours.\n",
  bindings: "# Bindings\n\nWhat each moment was about: people, things, places and ideas, named by their path in your life vault. Bindings point from the soul to the life vault, never the other way.\n",
  holds: "# Holds\n\nBeliefs, values and what you hold sacred or as duty, each as you stated it. Never inferred.\n",
  "of-me": "# Of me\n\nReadings of you: patterns, tensions and identities. Each names its elements and cites the moments whose confirmed readings show it, and stays a reading until you confirm it.\n",
  table: "# Table\n\nThe element table this soul uses, pinned: Memory (0), Image (1) and the sixteen elements in nine categories. Read only.\n",
};

const inside = (child: string, parent: string) => {
  const r = relative(parent, child);
  return r === "" || (!r.startsWith("..") && !isAbsolute(r));
};

async function filesIn(root: string, dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const d of await readdir(join(root, dir), { withFileTypes: true }).catch(() => [])) {
    const rel = join(dir, d.name);
    if (d.isDirectory()) out.push(...(await filesIn(root, rel)));
    else out.push(rel);
  }
  return out;
}

export type SoulInit = { soul: string; moved: string[] };

/**
 * Start a soul vault for the person whose life vault is at `lifeRoot`, and move the journal into it: the
 * life vault's journal/ becomes moments/ and its reflections/ become readings/. Files move whole and are
 * adopted into the soul vault's memory; the life vault's journal README is replaced by a note saying where
 * the journal went. Both reflect.json files name each other.
 */
export async function initSoul(lifeRoot: string, soulDir: string, now = new Date()): Promise<SoulInit> {
  const life = await readConfig(lifeRoot);
  if (life.layout !== "vault") throw new Error("a soul vault pairs with a life vault; this instance uses the older people/<id> layout");
  if (life.soul) throw new Error(`this vault already has a soul vault at ${resolve(lifeRoot, life.soul)}`);
  const soulRoot = resolve(soulDir);
  const absLife = resolve(lifeRoot);
  if (inside(soulRoot, absLife) || inside(absLife, soulRoot)) throw new Error("the soul vault is a folder of its own, neither inside the life vault nor holding it");
  if ((await readdir(soulRoot).catch(() => [])).length) throw new Error(`${soulRoot} is not empty`);
  const at = now.toISOString();
  const table = await readFile(new URL("./structure.json", import.meta.url), "utf8");
  const config: InstanceConfig = { reflect: 1, person: life.person, host: "files", encrypt: null, layout: "soul", life: relative(soulRoot, absLife) };
  const files: [string, string][] = [
    ["reflect.json", `${JSON.stringify(config, null, 2)}\n`],
    ["soul.json", `${JSON.stringify({ image: life.person, table: "table/structure.json", life: config.life }, null, 2)}\n`],
    ["table/structure.json", table],
    ...SOUL_DIRS.map((d): [string, string] => [join(d, "README.md"), README[d]]),
  ];
  let last = null;
  const lines = [];
  for (const [rel, bytes] of files) {
    await mkdir(join(soulRoot, rel, ".."), { recursive: true });
    await writeFile(join(soulRoot, rel), bytes);
    last = nextLine(last, "create", rel, bytes, at);
    lines.push(last);
  }
  await writeFile(join(soulRoot, "memory.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
  await writeFile(join(soulRoot, ".gitignore"), ".obsidian/workspace*.json\n.obsidian/cache\n.trash/\n.DS_Store\n.reflect/\n");

  // Move the journal: entries to moments/, reflections and labelings to readings/.
  const from = recordPaths("vault", life.person);
  const to = recordPaths("soul", life.person);
  const moved: string[] = [];
  for (const [src, dst] of [[from.journal, to.journal], [from.reflections, to.reflections]] as const) {
    for (const rel of await filesIn(absLife, src)) {
      const name = rel.slice(src.length + 1);
      if (name === "README.md" || name === ".gitkeep") continue;
      const target = join(dst, name);
      await mkdir(join(soulRoot, target, ".."), { recursive: true });
      await rename(join(absLife, rel), join(soulRoot, target));
      moved.push(target);
    }
    await rmdir(join(absLife, src)).catch(() => undefined);
  }
  if (moved.length) await adopt(soulRoot, moved, now);
  if ((await stat(join(absLife, from.journal)).catch(() => null))?.isDirectory()) {
    await writeRecord(absLife, join(from.journal, "README.md"), `# Journal\n\nYour journal is in your soul vault now, at ${relative(absLife, soulRoot)}/moments. Reflect reads and writes it there from this folder too.\n`, "append", at);
  }
  await writeRecord(absLife, "reflect.json", `${JSON.stringify({ ...life, soul: relative(absLife, soulRoot) }, null, 2)}\n`, "append", at);
  return { soul: soulRoot, moved };
}

export type SoulStatus = { root: string; moments: number; readings: number; bindings: number; holds: number; ofMe: number };

export async function soulStatus(soulRoot: string): Promise<SoulStatus> {
  const count = async (dir: string) => (await filesIn(soulRoot, dir)).filter((f) => !f.endsWith("README.md") && !f.endsWith(".gitkeep") && !f.endsWith(".draft.json")).length;
  return { root: soulRoot, moments: await count("moments"), readings: await count("readings"), bindings: await count("bindings"), holds: await count("holds"), ofMe: await count("of-me") };
}

/** Problems for `reflect check` in a soul vault: soul.json names this person, and the life vault it names links back. */
export async function soulProblems(soulRoot: string, config: InstanceConfig): Promise<string[]> {
  const out: string[] = [];
  const soul = await readRecord(soulRoot, "soul.json");
  if (soul === null) out.push("soul.json is missing");
  else if ((JSON.parse(soul) as { image?: string }).image !== config.person) out.push("soul.json names another person");
  if (!config.life) out.push("reflect.json names no life vault");
  else {
    const lifeRoot = resolve(soulRoot, config.life);
    const life = await readConfig(lifeRoot).catch(() => null);
    if (!life) out.push(`no life vault at ${lifeRoot}`);
    else if (life.person !== config.person || !life.soul || resolve(lifeRoot, life.soul) !== resolve(soulRoot)) out.push(`the life vault at ${lifeRoot} does not name this soul vault`);
  }
  return out;
}
