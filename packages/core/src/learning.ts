// What a person knows, needs to know and wants to know, as files in learn/. A topic is a note in
// learn/topics/; what the person wants of it (aspire, or need for something they name) is a field, and
// what they have shown of it is evidence, kept as dated lines the way evidence-centered design (Mislevy,
// Steinberg and Almond 2003) backs a claim. Learned is derived from evidence, never set. What is
// possible to learn next is derived too: the fringe of knowledge space theory (Doignon and Falmagne),
// a topic whose prerequisites are all learned. Topics point at a shared map (Wikidata, ESCO) instead of
// Reflect keeping one. A resource is a note in learn/resources/: a link to a channel, article, book or
// course, and finishing one counts as studying the topics it names.
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { headOf } from "./lifetime.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const TOPICS = join("learn", "topics");
export const RESOURCES = join("learn", "resources");

export const WANTS = ["aspire", "need"] as const;
export type Want = (typeof WANTS)[number];

/** Kinds of evidence. `said` alone leaves a topic stated; any other kind shows it. */
export const EVIDENCE_KINDS = [
  { id: "said", text: "you say you know it" },
  { id: "studied", text: "you finished a book, course or lecture on it" },
  { id: "practiced", text: "you used it in real work; link the record in work/ or a venture" },
  { id: "made", text: "something others can inspect: a repo, an essay, a build" },
  { id: "assessed", text: "an exam or test result" },
  { id: "certified", text: "a credential; link its card in documents/" },
  { id: "vouched", text: "someone who knows the subject says you know it (--by)" },
  { id: "taught", text: "you taught it to others" },
] as const;
export type EvidenceKind = (typeof EVIDENCE_KINDS)[number]["id"];
export const isEvidenceKind = (s: string): s is EvidenceKind => EVIDENCE_KINDS.some((k) => k.id === s);

export const RESOURCE_KINDS = ["video", "channel", "article", "book", "course", "podcast", "paper", "lab", "site"] as const;
export const RESOURCE_STATUS = ["saved", "started", "finished", "dropped"] as const;
export type ResourceStatus = (typeof RESOURCE_STATUS)[number];

/** A topic's id in a shared map: a Wikidata item, an ESCO concept, or a link. */
const SAME_AS = /^(wikidata:Q\d+|esco:\S+|https?:\/\/\S+)$/;

export type Evidence = { on: string; kind: EvidenceKind; link?: string; by?: string; note?: string; from?: string };
export type Topic = { id: string; title: string; want?: Want; for?: string; same_as?: string; requires: string[]; part_of: string[]; evidence: Evidence[] };
export type Resource = { id: string; kind: string; title: string; url?: string; by?: string; platform?: string; topics: string[]; status: ResourceStatus };

export function slugOf(name: string): string {
  const s = name.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
  if (!s) throw new Error("a topic or resource needs a name with letters or digits");
  return s;
}

const links = (v: string | undefined) => [...(v ?? "").matchAll(/\[\[(?:learn\/topics\/)?([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map((m) => m[1]!.trim());
const linkList = (ids: readonly string[]) => ids.map((t) => `[[learn/topics/${t}]]`).join(", ");
const q = (s: string) => JSON.stringify(s);
const oneLine = (s: string) => s.replace(/\s+/g, " ").trim();

function body(text: string): { head: Record<string, string>; history: string[] } {
  const t = text.replace(/\r\n/g, "\n");
  const end = t.startsWith("---\n") ? t.indexOf("\n---\n", 4) : -1;
  const rest = end > 0 ? t.slice(end + 5) : t;
  return { head: headOf(t), history: rest.split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2)) };
}

/** An evidence line: `2026-11-11 made: https://… by Ana: a note`. Other history lines are not evidence. */
export function parseEvidence(line: string): Evidence | null {
  const m = /^(\d{4}-\d{2}(?:-\d{2})?) evidence ([a-z]+)(?: <([^>]*)>)?(?: by ([^:]+?))?(?:: (.*))?$/.exec(line);
  if (!m || !isEvidenceKind(m[2]!)) return null;
  return { on: m[1]!, kind: m[2], ...(m[3] ? { link: m[3] } : {}), ...(m[4] ? { by: m[4] } : {}), ...(m[5] ? { note: m[5] } : {}) };
}

function evidenceLine(e: Evidence): string {
  return `${e.on} evidence ${e.kind}${e.link ? ` <${e.link}>` : ""}${e.by ? ` by ${oneLine(e.by)}` : ""}${e.note ? `: ${oneLine(e.note)}` : ""}`;
}

export function parseTopic(id: string, text: string): Topic {
  const { head, history } = body(text);
  const want = (WANTS as readonly string[]).includes(head.want ?? "") ? (head.want as Want) : undefined;
  return {
    id,
    title: head.title ?? id,
    ...(want ? { want } : {}),
    ...(head.for ? { for: head.for } : {}),
    ...(head.same_as ? { same_as: head.same_as } : {}),
    requires: links(head.requires),
    part_of: links(head.part_of),
    evidence: history.map(parseEvidence).filter((e): e is Evidence => e !== null),
  };
}

function topicText(t: Topic, history: readonly string[]): string {
  const head = [`topic: ${t.id}`, `title: ${q(t.title)}`];
  if (t.want) head.push(`want: ${t.want}`);
  if (t.for) head.push(`for: ${q(t.for)}`);
  if (t.same_as) head.push(`same_as: ${t.same_as}`);
  if (t.requires.length) head.push(`requires: ${linkList(t.requires)}`);
  if (t.part_of.length) head.push(`part_of: ${linkList(t.part_of)}`);
  return `---\n${head.join("\n")}\n---\n\n# ${t.title}\n\n${history.map((h) => `- ${h}`).join("\n")}\n`;
}

export function parseResource(id: string, text: string): Resource {
  const { head } = body(text);
  return {
    id,
    kind: head.kind ?? "",
    title: head.title ?? id,
    ...(head.url ? { url: head.url } : {}),
    ...(head.by ? { by: head.by } : {}),
    ...(head.platform ? { platform: head.platform } : {}),
    topics: links(head.topics),
    status: (RESOURCE_STATUS as readonly string[]).includes(head.status ?? "") ? (head.status as ResourceStatus) : "saved",
  };
}

function resourceText(r: Resource, history: readonly string[]): string {
  const head = [`resource: ${r.id}`, `kind: ${r.kind}`, `title: ${q(r.title)}`];
  if (r.url) head.push(`url: ${r.url}`);
  if (r.by) head.push(`by: ${q(r.by)}`);
  if (r.platform) head.push(`platform: ${r.platform}`);
  if (r.topics.length) head.push(`topics: ${linkList(r.topics)}`);
  head.push(`status: ${r.status}`);
  return `---\n${head.join("\n")}\n---\n\n# ${r.title}\n\n${r.url ? `${r.url}\n\n` : ""}${history.map((h) => `- ${h}`).join("\n")}\n`;
}

export type TopicChange = { title?: string; want?: Want; for?: string; same_as?: string; requires?: string[]; part_of?: string[] };

/** Keep what the person wants of a topic, and where it sits. Each change adds a dated line. */
export async function keepTopic(root: string, name: string, change: TopicChange, now: Date, evidence?: Omit<Evidence, "on"> & { on?: string }): Promise<Topic> {
  const id = slugOf(name);
  if (change.want === "need" && !change.for) throw new Error("a need names what it is for: an intent, a venture, a need or a role (--for)");
  if (change.same_as && !SAME_AS.test(change.same_as)) throw new Error("--same-as is wikidata:Q<number>, esco:<uri> or a link");
  if (evidence) evidenceProblem(evidence);
  const rel = join(TOPICS, `${id}.md`);
  const old = await readRecord(root, rel);
  const prev = old ? parseTopic(id, old) : { id, title: name.trim(), requires: [], part_of: [], evidence: [] };
  const t: Topic = {
    ...prev,
    ...(change.title ? { title: change.title } : {}),
    ...(change.want ? { want: change.want, ...(change.want === "need" ? { for: change.for } : {}) } : {}),
    ...(change.same_as ? { same_as: change.same_as } : {}),
    requires: [...new Set([...prev.requires, ...(change.requires ?? []).map(slugOf)])],
    part_of: [...new Set([...prev.part_of, ...(change.part_of ?? []).map(slugOf)])],
  };
  if (change.want === "aspire") delete t.for;
  if (t.requires.includes(id) || t.part_of.includes(id)) throw new Error("a topic cannot require itself or be part of itself");
  const at = now.toISOString();
  const day = at.slice(0, 10);
  const lines = old ? body(old).history : [];
  if (change.want) lines.push(`${day} want ${change.want}${change.want === "need" ? ` for ${oneLine(change.for!)}` : ""}`);
  if (evidence) {
    const e = { ...evidence, on: evidence.on ?? day };
    lines.push(evidenceLine(e));
    t.evidence = [...t.evidence, e];
  }
  if (!old && !lines.length) lines.push(`${day} noted`);
  await writeRecord(root, rel, topicText(t, lines), old ? "append" : "create", at);
  return t;
}

function evidenceProblem(e: Partial<Evidence>): void {
  if (!e.kind || !isEvidenceKind(e.kind)) throw new Error(`evidence is one of ${EVIDENCE_KINDS.map((k) => k.id).join(", ")}`);
  if (e.on && !/^\d{4}-\d{2}(-\d{2})?$/.test(e.on)) throw new Error("--on is YYYY-MM or YYYY-MM-DD");
  if (e.kind === "vouched" && !e.by) throw new Error("vouched evidence names who vouches (--by)");
  if (e.kind !== "said" && !e.link && !e.note) throw new Error(`${e.kind} evidence says where it can be seen (--link or --note)`);
  if (e.link && /[<>\s]/.test(e.link)) throw new Error("--link has no spaces or angle brackets");
}

export type ResourceChange = { kind?: string; title?: string; url?: string; by?: string; platform?: string; topics?: string[]; status?: ResourceStatus };

/** Keep a resource: a link the person saved, and how far they are with it. */
export async function keepResource(root: string, name: string, change: ResourceChange, now: Date): Promise<Resource> {
  const id = slugOf(name);
  const rel = join(RESOURCES, `${id}.md`);
  const old = await readRecord(root, rel);
  if (!old && !change.kind) throw new Error(`a new resource needs --kind: ${RESOURCE_KINDS.join(", ")}`);
  if (change.kind && !(RESOURCE_KINDS as readonly string[]).includes(change.kind)) throw new Error(`--kind is one of ${RESOURCE_KINDS.join(", ")}`);
  if (change.status && !(RESOURCE_STATUS as readonly string[]).includes(change.status)) throw new Error(`--status is one of ${RESOURCE_STATUS.join(", ")}`);
  if (change.url && !/^(https?:\/\/\S+|isbn:[0-9X-]+|doi:\S+)$/i.test(change.url)) throw new Error("--url is a link, isbn:<number> or doi:<id>");
  if (change.platform && !(await platformIds(root)).includes(change.platform)) throw new Error(`no platform ${change.platform}; connect it first: reflect connect ${change.platform} --name <name> --kind platform`);
  const prev = old ? parseResource(id, old) : { id, kind: change.kind!, title: name.trim(), topics: [], status: "saved" as const };
  const r: Resource = {
    ...prev,
    ...(change.kind ? { kind: change.kind } : {}),
    ...(change.title ? { title: change.title } : {}),
    ...(change.url ? { url: change.url } : {}),
    ...(change.by ? { by: change.by } : {}),
    ...(change.platform ? { platform: change.platform } : {}),
    ...(change.status ? { status: change.status } : {}),
    topics: [...new Set([...prev.topics, ...(change.topics ?? []).map(slugOf)])],
  };
  const at = now.toISOString();
  const lines = old ? body(old).history : [];
  if (!old || (change.status && change.status !== prev.status)) lines.push(`${at.slice(0, 10)} ${r.status}`);
  await writeRecord(root, rel, resourceText(r, lines), old ? "append" : "create", at);
  return r;
}

async function notes(root: string, dir: string): Promise<[string, string][]> {
  const names = (await readdir(join(root, dir)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md").sort();
  return Promise.all(names.map(async (n) => [n.slice(0, -3), await readFile(join(root, dir, n), "utf8")] as [string, string]));
}

async function platformIds(root: string): Promise<string[]> {
  const names = (await readdir(join(root, "connections")).catch(() => [] as string[])).filter((n) => n.endsWith(".json"));
  const out: string[] = [];
  for (const n of names) {
    try {
      const c = JSON.parse(await readFile(join(root, "connections", n), "utf8")) as { id?: string; kind?: string };
      if (c.kind === "platform" && c.id) out.push(c.id);
    } catch {
      // documentProblems reports it
    }
  }
  return out;
}

export type Standing = "stated" | "shown";
export type MapRow = { topic: Topic; evidence: Evidence[]; standing?: Standing; resources: Resource[]; blockedBy: string[] };
export type LearnMap = { learned: MapRow[]; aspire: MapRow[]; need: MapRow[]; possible: MapRow[] };

/**
 * The person's map: every topic with its evidence, including studying from finished resources. Learned is
 * a topic with evidence; aspire and need are the rest that the person wants; possible is the fringe, a
 * topic they have not learned or asked for whose prerequisites are all learned, and which requires
 * something or is required by something on the map.
 */
export async function learnMap(root: string): Promise<LearnMap> {
  const resources = (await notes(root, RESOURCES)).map(([id, t]) => parseResource(id, t));
  const topics = new Map((await notes(root, TOPICS)).map(([id, t]) => [id, parseTopic(id, t)] as const));
  for (const t of topics.values()) for (const r of [...t.requires, ...t.part_of]) if (!topics.has(r)) topics.set(r, { id: r, title: r, requires: [], part_of: [], evidence: [] });
  for (const r of resources) for (const id of r.topics) if (!topics.has(id)) topics.set(id, { id, title: id, requires: [], part_of: [], evidence: [] });
  const rows = new Map<string, MapRow>();
  for (const t of topics.values()) {
    const studied = resources.filter((r) => r.status === "finished" && r.topics.includes(t.id)).map((r): Evidence => ({ on: "", kind: "studied", from: `learn/resources/${r.id}`, ...(r.url ? { link: r.url } : {}) }));
    const evidence = [...t.evidence, ...studied];
    const standing: Standing | undefined = evidence.length ? (evidence.some((e) => e.kind !== "said") ? "shown" : "stated") : undefined;
    rows.set(t.id, { topic: t, evidence, ...(standing ? { standing } : {}), resources: resources.filter((r) => r.topics.includes(t.id)), blockedBy: [] });
  }
  for (const row of rows.values()) row.blockedBy = row.topic.requires.filter((r) => !rows.get(r)?.standing);
  const all = [...rows.values()];
  return {
    learned: all.filter((r) => r.standing),
    aspire: all.filter((r) => !r.standing && r.topic.want === "aspire"),
    need: all.filter((r) => !r.standing && r.topic.want === "need"),
    possible: all.filter((r) => !r.standing && !r.topic.want && r.blockedBy.length === 0 && (r.topic.requires.length > 0 || all.some((o) => o.topic.requires.includes(r.topic.id)))),
  };
}

/** Problems in learn/ for `reflect check`. */
export async function learnProblems(root: string): Promise<string[]> {
  const out: string[] = [];
  const platforms = await platformIds(root);
  for (const [id, text] of await notes(root, TOPICS)) {
    const f = `learn/topics/${id}.md`;
    const { head, history } = body(text);
    if (head.topic !== id) out.push(`${f}: topic must match the file name`);
    if (head.want && !(WANTS as readonly string[]).includes(head.want)) out.push(`${f}: want is aspire or need`);
    if (head.want === "need" && !head.for) out.push(`${f}: a need names what it is for`);
    if (head.same_as && !SAME_AS.test(head.same_as)) out.push(`${f}: same_as is wikidata:Q<number>, esco:<uri> or a link`);
    for (const l of history) if (/^\S+ evidence /.test(l) && !parseEvidence(l)) out.push(`${f}: evidence line not understood: ${l}`);
  }
  const topics = new Map((await notes(root, TOPICS)).map(([id, t]) => [id, parseTopic(id, t)] as const));
  const seen = new Set<string>();
  const cycle = (id: string, path: string[]): string | null => {
    if (path.includes(id)) return [...path.slice(path.indexOf(id)), id].join(" requires ");
    if (seen.has(id)) return null;
    for (const r of topics.get(id)?.requires ?? []) {
      const c = cycle(r, [...path, id]);
      if (c) return c;
    }
    seen.add(id);
    return null;
  };
  for (const id of topics.keys()) {
    const c = cycle(id, []);
    if (c) {
      out.push(`learn/topics: prerequisites go round in a circle: ${c}`);
      break;
    }
  }
  for (const [id, text] of await notes(root, RESOURCES)) {
    const f = `learn/resources/${id}.md`;
    const { head } = body(text);
    if (head.resource !== id) out.push(`${f}: resource must match the file name`);
    if (!(RESOURCE_KINDS as readonly string[]).includes(head.kind ?? "")) out.push(`${f}: kind is one of ${RESOURCE_KINDS.join(", ")}`);
    if (!(RESOURCE_STATUS as readonly string[]).includes(head.status ?? "")) out.push(`${f}: status is one of ${RESOURCE_STATUS.join(", ")}`);
    if (head.platform && !platforms.includes(head.platform)) out.push(`${f}: names a platform that is not connected: ${head.platform}`);
  }
  return out;
}
