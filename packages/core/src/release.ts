// The release plan: what of the legacy folder goes to whom after the person's death, and who may confirm
// the death. The plan is a record the person keeps in legacy/release.json. A hard rule holds whatever it
// says: only what the person made to survive them can be given, which is the files under legacy/ and the
// journal entries marked `legacy: true`. The rest of the journal, reflections and everything else is
// never released. The Reflect app carries a release out (web/src/server/legacy-store.ts); `previewRelease`
// is what it is sent for each person.
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { headExtras } from "./files-journal.ts";
import type { RecordPaths } from "./layout.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const PLAN_PATH = join("legacy", "release.json");
/** A gift of every journal entry marked to survive the person. */
export const LEGACY_ENTRIES = "legacy-entries";

export type Trustee = { name: string; contact: string };
export type Gift = { what: string; to: string; contact: string };
export type ReleasePlan = { release: 1; trustees: Trustee[]; confirmations: number; wait_days: number; gifts: Gift[] };

export const emptyPlan = (): ReleasePlan => ({ release: 1, trustees: [], confirmations: 2, wait_days: 30, gifts: [] });

export async function readPlan(root: string): Promise<ReleasePlan> {
  const t = await readRecord(root, PLAN_PATH);
  return t ? (JSON.parse(t) as ReleasePlan) : emptyPlan();
}

/** Problems that make a plan wrong, and notes on what it leaves out. */
export async function planProblems(root: string, plan: ReleasePlan): Promise<{ problems: string[]; notes: string[] }> {
  const problems: string[] = [];
  const notes: string[] = [];
  if (plan.release !== 1) problems.push("legacy/release.json: not a release plan");
  const trustees = Array.isArray(plan.trustees) ? plan.trustees : [];
  for (const t of trustees) if (!t?.name || !t?.contact) problems.push("legacy/release.json: each trustee needs a name and a contact");
  if (!Number.isInteger(plan.confirmations) || plan.confirmations < 1) problems.push("legacy/release.json: confirmations is a whole number, 1 or more");
  else if (trustees.length && plan.confirmations > trustees.length) notes.push(`Nothing can be released yet: the plan asks ${plan.confirmations} confirmations of ${trustees.length} trustee(s); name another, or set fewer.`);
  if (!Number.isInteger(plan.wait_days) || plan.wait_days < 0) problems.push("legacy/release.json: wait_days is a whole number of days, 0 or more");
  for (const g of Array.isArray(plan.gifts) ? plan.gifts : []) {
    if (!g?.to || !g?.contact) problems.push(`legacy/release.json: the gift of ${g?.what} needs a person and a contact`);
    const why = giftRefusal(g?.what);
    if (why) problems.push(`legacy/release.json: ${why}`);
    else if (g.what !== LEGACY_ENTRIES && (await readRecord(root, g.what)) === null) problems.push(`legacy/release.json: ${g.what} does not exist`);
  }
  if (plan.gifts?.length && !trustees.length) notes.push("Nothing can be released yet: name at least one trustee who may confirm your death.");
  if (trustees.length && plan.confirmations === 1 && trustees.length > 1) notes.push("One trustee alone can confirm your death; two is safer when you have them.");
  const letters = await readdir(join(root, "legacy", "letters")).catch(() => [] as string[]);
  for (const l of letters.filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md")) {
    if (!plan.gifts?.some((g) => g.what === join("legacy", "letters", l))) notes.push(`legacy/letters/${l} is given to no one yet.`);
  }
  return { problems, notes };
}

/** Why something can never be given, or null when it can. */
export function giftRefusal(what: unknown): string | null {
  if (typeof what !== "string" || !what) return "a gift names what is given";
  if (what === LEGACY_ENTRIES) return null;
  const norm = what.replace(/\\/g, "/");
  if (norm.includes("..") || norm.startsWith("/")) return `${what} is outside the vault`;
  if (!norm.startsWith("legacy/")) return `${what} can never be released: only the legacy folder and entries marked to survive you can`;
  if (norm === PLAN_PATH) return "the release plan itself is not a gift";
  return null;
}

export async function addTrustee(root: string, name: string, contact: string, now: Date): Promise<ReleasePlan> {
  const plan = await readPlan(root);
  const trustees = [...plan.trustees.filter((t) => t.name !== name), { name, contact }];
  return savePlan(root, { ...plan, trustees }, now);
}

export async function addGift(root: string, what: string, to: string, contact: string, now: Date): Promise<ReleasePlan> {
  const why = giftRefusal(what);
  if (why) throw new Error(why);
  if (what !== LEGACY_ENTRIES && (await readRecord(root, what)) === null) throw new Error(`${what} does not exist`);
  const plan = await readPlan(root);
  const gifts = [...plan.gifts.filter((g) => !(g.what === what && g.to === to)), { what, to, contact }];
  return savePlan(root, { ...plan, gifts }, now);
}

export async function setTerms(root: string, terms: { confirmations?: number; wait_days?: number }, now: Date): Promise<ReleasePlan> {
  const plan = await readPlan(root);
  return savePlan(root, { ...plan, ...terms }, now);
}

async function savePlan(root: string, plan: ReleasePlan, now: Date): Promise<ReleasePlan> {
  const { problems } = await planProblems(root, plan);
  if (problems.length) throw new Error(problems.join("; "));
  const old = await readRecord(root, PLAN_PATH);
  await writeRecord(root, PLAN_PATH, `${JSON.stringify(plan, null, 2)}\n`, old ? "append" : "create", now.toISOString());
  return plan;
}

/** The journal entries marked to survive the person. */
export async function legacyEntries(root: string, paths: RecordPaths): Promise<string[]> {
  const names = await readdir(join(root, paths.journal)).catch(() => [] as string[]);
  const out: string[] = [];
  for (const n of names.filter((x) => x.endsWith(".md") && x.toLowerCase() !== "readme.md").sort()) {
    const rel = join(paths.journal, n);
    if (headExtras(await readFile(join(root, rel), "utf8")).includes("legacy: true")) out.push(rel);
  }
  return out;
}

/** What a release would hand each person, by name: the files, with legacy entries expanded. */
export async function previewRelease(root: string, paths: RecordPaths, plan: ReleasePlan): Promise<Map<string, string[]>> {
  const out = new Map<string, string[]>();
  const entries = await legacyEntries(root, paths);
  for (const g of plan.gifts) {
    if (giftRefusal(g.what)) continue;
    const files = g.what === LEGACY_ENTRIES ? entries : [g.what];
    out.set(g.to, [...new Set([...(out.get(g.to) ?? []), ...files])]);
  }
  return out;
}
