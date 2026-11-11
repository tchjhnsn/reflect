// Where the journal lives. A life vault linked to a soul vault (soul.ts) keeps no journal of its own: its
// entries are moments in the soul vault. Commands that read or write entries ask here first, so they work
// the same from either folder. The soul vault reads the life vault's sources (pointer entries) and keeps
// concept notes there, since a concept is a thing in shared reality, not a moment.
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { recordPaths, type Layout, type RecordPaths } from "./layout.ts";

type Link = { person: string; layout?: Layout; soul?: string; life?: string };

async function link(root: string): Promise<Link | null> {
  try {
    return JSON.parse(await readFile(join(root, "reflect.json"), "utf8")) as Link;
  } catch {
    return null;
  }
}

export type JournalHome = {
  /** The folder whose memory.jsonl the journal writes to. */
  root: string;
  layout: Layout;
  paths: RecordPaths;
  /** The life vault: where sources.json and concept notes are. */
  lifeRoot: string;
  /** True when the journal is in a soul vault. */
  soul: boolean;
};

/** The journal's folder for a vault: its own, or the soul vault it links to. */
export async function journalHome(root: string): Promise<JournalHome> {
  const c = await link(root);
  if (!c) throw new Error(`no reflect.json in ${root}`);
  if (c.layout === "soul") return { root, layout: "soul", paths: recordPaths("soul", c.person), lifeRoot: c.life ? resolve(root, c.life) : root, soul: true };
  if (c.soul) {
    const soulRoot = resolve(root, c.soul);
    const s = await link(soulRoot);
    if (!s || s.layout !== "soul" || s.person !== c.person) throw new Error(`reflect.json names a soul vault at ${soulRoot}, but there is none for this person there`);
    return { root: soulRoot, layout: "soul", paths: recordPaths("soul", c.person), lifeRoot: root, soul: true };
  }
  const layout: Layout = c.layout ?? "people";
  return { root, layout, paths: recordPaths(layout, c.person), lifeRoot: root, soul: false };
}
