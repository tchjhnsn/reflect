// The Ventures of a Lifetime as folders. Folders are the statechart's regions and the body politic's
// associations, which last a whole life; states are never folders. A state is a field, and each move
// between states is a dated passage the person keeps (life/passages.jsonl), the way Naive derives
// "founder" from facts instead of filing anyone under it. life/state.json is the latest passage in each
// region, rebuilt from the passages, never written by hand.
import { join } from "node:path";
import type { RecordPaths } from "./layout.ts";
import { readRecord, writeRecord } from "./record-io.ts";

/** The folders of a vault, each with the guide its README carries. Order is the order of a life. */
export const LIFETIME_FOLDERS: readonly { path: string; title: string; guide: string }[] = [
  { path: "life", title: "Life", guide: "Where you are in each region of your life. `reflect passage` keeps a move from one state to another, dated; `state.json` is the latest in each region and is rebuilt from `passages.jsonl`." },
  { path: "needs", title: "Needs", guide: "Self-preservation, one file per need: food, water, shelter, health, transport, income, knowledge, safety. Each says how the need is met now: self-provided, stocked, supplied or cared for, and by whom. `reflect need` keeps it." },
  { path: "learn", title: "Learn", guide: "What you know, by your own account. Books are the center: what you have read, and how much of it you kept. `reflect learn` shows what you know, your books and what of them you have not rated, what you aspire to and need to learn, and what you could learn next. Nothing here is known until you say how well you grasp it." },
  { path: "learn/books", title: "Books", guide: "A note per book: title, authors, isbn, status (want, reading, read, abandoned) with dates, and the subjects it covers. When you finish one, try to name its key subjects without looking, then rate your grasp of each." },
  { path: "learn/subjects", title: "Subjects", guide: "A note per subject, a node in your knowledge graph. Each grasp line is your own rating, dated, often from a book: teach (could teach it), explain (could explain the key ideas), gist (remember the gist) or forgot. `want` is aspire, or need with `for` naming what needs it. `requires` and `part_of` link other subjects; `same_as` ties it to the shared map (wikidata:Q<number>, lcsh:sh<number> or esco:<uri>). A subject is an input, a field you study; a framing of your own belongs in ideas/. `shown by` lines are evidence you displayed knowledge of it and `suggested by` lines say someone thinks it relates to you; neither is a grasp." },
  { path: "learn/resources", title: "Resources", guide: "A note per link you keep for what you want to learn: a YouTube channel or video, an article, a post or thread on X, Substack or Reddit, a newsletter, a course, a podcast, a paper, a lab or a site, with the subjects it serves (its `source` is read from the link) and how far you are (saved, started, finished, dropped). A course on Khan Academy, Udemy, Coursera or Fanout names its platform, a connection of kind platform. Books go in learn/books." },
  { path: "learn/films", title: "Films", guide: "A note per film, documentary, series or short: what it is, whether you watched it (want, watching, watched, abandoned, or referenced when you cited it without saying you watched it), and the subjects it covers." },
  { path: "learn/talks", title: "Talks", guide: "A note per talk you heard: a lecture, a sermon, a keynote, a debate, a video of someone speaking. Who spoke, when, where, and the subjects it covers." },
  { path: "learn/conversations", title: "Conversations", guide: "A note per conversation that fed your thinking: with whom, when, and what about, in your words. Their words stay theirs." },
  { path: "ideas", title: "Ideas", guide: "Your own ideas, the outputs of your thinking, kept apart from what you take in (learn/). Each note says the idea in your words, dated each time you restate it, and links its inputs: `from` what you say fed it, `maybe from` what you suspect did. Recommendations are a separate list, each placed from supports to opposes, with who suggested it; one becomes an input only when you say it fed the idea. `reflect idea` keeps them." },
  { path: "work", title: "Work", guide: "Work and provide. Each venture you build is a folder here, a full Naive venture with its own provenance, so Reflect's check leaves it to Naive." },
  { path: "belonging/friends", title: "Friends", guide: "A note per friend: who they are to you, and what it means to you to be one." },
  { path: "belonging/groups", title: "Groups", guide: "The groups you belong to: teams, clubs, coalitions, a congregation." },
  { path: "belonging/standing", title: "Standing", guide: "Offices you hold, roles you are trusted with, the reputation you are building." },
  { path: "line", title: "Line", guide: "Continuing the line: seeking a mate, a partner, raising children, caring for kin. `partner.md` is yours to keep." },
  { path: "household/members", title: "Members", guide: "A folder per member of your household. A child's records live here while they are in your care; once they are past the threshold your community sets, their folder can become a vault of their own." },
  { path: "household/home", title: "Home", guide: "The home itself: where, lease or deed, upkeep." },
  { path: "household/money", title: "Money", guide: "The household's budget, accounts and plans. Statements stay with the bank; a connection can say so." },
  { path: "community", title: "Community", guide: "The village: neighbors, congregation, local groups, and what you give to them." },
  { path: "polis", title: "Polis", guide: "Civic life: your districts, offices, votes and the questions you care about. Polity connects here, and only with a grant." },
  { path: "legacy", title: "Legacy", guide: "What you prepare for your death and what you want to survive you: `wishes.md`, `bucket-list.md`, letters to people, and journal entries you mark to survive you (`reflect legacy`). The will, life insurance and the rest are on the documents checklist." },
  { path: "legacy/letters", title: "Letters", guide: "A letter per person, to be given to them after your death. Nothing here is released before then, and only as you say." },
];

/** Files a vault starts with beside the READMEs: templates the person fills in, in their words. */
export const LEGACY_TEMPLATES: readonly [string, string][] = [
  ["legacy/wishes.md", "# Wishes\n\nWhat you want done when you die. Leave any part blank.\n\n## Service\n\n## Body: burial, cremation, donation\n\n## Organ donation\n\n## Who should be told first\n\n## Music, readings, words\n"],
  ["legacy/bucket-list.md", "# Bucket list\n\nWhat you want to do, see or make before you die. Check one off when it is done; one you mean to start can become an intent.\n\n- [ ] \n"],
];

/** The needs of self-preservation (adopted 2026-11-09), each met in one of four ways. */
export const NEEDS = ["food", "water", "shelter", "health", "transport", "income", "knowledge", "safety"] as const;
export type Need = (typeof NEEDS)[number];
export const NEED_STATES = ["self-provided", "stocked", "supplied", "cared-for"] as const;
export type NeedState = (typeof NEED_STATES)[number];
export const PROVIDERS = ["self", "market", "nature", "polis", "kin"] as const;

/** The regions of adult life and their states, from the Ventures of a Lifetime statechart. */
export const REGIONS = {
  provide: ["learning", "working"],
  belonging: ["affiliating", "gaining-status"],
  line: ["seeking-a-mate", "pair-bond", "raising-children", "caring-for-kin"],
} as const;
export type Region = keyof typeof REGIONS;
export const isRegion = (s: string): s is Region => Object.hasOwn(REGIONS, s);

export type Passage = { region: Region; to: string; on: string; note?: string; at: string };

function frontMatter(head: string[], title: string, history: readonly string[]): string {
  return `---\n${head.join("\n")}\n---\n\n# ${title}\n\n${history.map((h) => `- ${h}`).join("\n")}\n`;
}

function historyOf(text: string | null): string[] {
  if (!text) return [];
  const body = text.replace(/\r\n/g, "\n");
  const end = body.startsWith("---\n") ? body.indexOf("\n---\n", 4) : -1;
  return (end > 0 ? body.slice(end + 5) : body).split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2));
}

export function headOf(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  const body = text.replace(/\r\n/g, "\n");
  if (!body.startsWith("---\n")) return out;
  const end = body.indexOf("\n---\n", 4);
  for (const line of body.slice(4, end > 0 ? end : 0).split("\n")) {
    const m = /^([a-z_]+):\s*(.*)$/.exec(line);
    if (m && m[2]) out[m[1]!] = m[2].startsWith('"') ? (JSON.parse(m[2]) as string) : m[2];
  }
  return out;
}

/** Keep how one need is met now. Each change adds a line; earlier ones stay. */
export async function keepNeed(root: string, need: string, state: string, opts: { provider?: string; runway?: string; note?: string }, now: Date): Promise<string> {
  if (!(NEEDS as readonly string[]).includes(need)) throw new Error(`a need is one of ${NEEDS.join(", ")}`);
  if (!(NEED_STATES as readonly string[]).includes(state)) throw new Error(`a need is met one of these ways: ${NEED_STATES.join(", ")}`);
  if (opts.provider && !(PROVIDERS as readonly string[]).includes(opts.provider)) throw new Error(`--provider is one of ${PROVIDERS.join(", ")}`);
  if (state === "stocked" && !opts.runway) throw new Error("stocked needs --runway: how long the stock lasts, like 30d or 6m");
  if (opts.runway && !/^\d+[dwmy]$/.test(opts.runway)) throw new Error("--runway is a number and d, w, m or y, like 30d");
  const rel = join("needs", `${need}.md`);
  const old = await readRecord(root, rel);
  const at = now.toISOString();
  const head = [`need: ${need}`, `state: ${state}`];
  if (opts.provider) head.push(`provider: ${opts.provider}`);
  if (opts.runway) head.push(`runway: ${opts.runway}`);
  head.push(`kept_at: ${at}`);
  const line = `${at.slice(0, 10)} ${state}${opts.provider ? ` by ${opts.provider}` : ""}${opts.runway ? ` for ${opts.runway}` : ""}${opts.note ? `: ${opts.note.replace(/\n/g, " ")}` : ""}`;
  await writeRecord(root, rel, frontMatter(head, need[0]!.toUpperCase() + need.slice(1), [...historyOf(old), line]), old ? "append" : "create", at);
  return rel;
}

export function parsePassages(text: string | null): Passage[] {
  return (text ?? "").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Passage);
}

/** The latest passage in each region, by the date it happened, then when it was kept. */
export function lifeState(passages: readonly Passage[]): Partial<Record<Region, { state: string; since: string }>> {
  const out: Partial<Record<Region, { state: string; since: string }>> = {};
  const sorted = [...passages].sort((a, b) => a.on.localeCompare(b.on) || a.at.localeCompare(b.at));
  for (const p of sorted) out[p.region] = { state: p.to, since: p.on };
  return out;
}

/** Keep a passage the person states, and rebuild state.json from every passage. */
export async function keepPassage(root: string, region: string, to: string, on: string, note: string | undefined, now: Date): Promise<Passage> {
  if (!isRegion(region)) throw new Error(`a region is one of ${Object.keys(REGIONS).join(", ")}`);
  if (!(REGIONS[region] as readonly string[]).includes(to)) throw new Error(`in ${region} a state is one of ${REGIONS[region].join(", ")}`);
  if (!/^\d{4}-\d{2}(-\d{2})?$/.test(on)) throw new Error("--on is YYYY-MM or YYYY-MM-DD");
  const at = now.toISOString();
  const p: Passage = { region, to, on, ...(note ? { note } : {}), at };
  const rel = join("life", "passages.jsonl");
  const old = await readRecord(root, rel);
  const passages = [...parsePassages(old), p];
  await writeRecord(root, rel, `${passages.map((x) => JSON.stringify(x)).join("\n")}\n`, old ? "append" : "create", at);
  const stateRel = join("life", "state.json");
  const hadState = (await readRecord(root, stateRel)) !== null;
  await writeRecord(root, stateRel, `${JSON.stringify(lifeState(passages), null, 2)}\n`, hadState ? "append" : "create", at);
  return p;
}

/**
 * Prompts for what the person wants to survive them. The first five follow the Dignity Therapy question
 * protocol (Chochinov et al., Journal of Clinical Oncology 23(24), 2005), the rest the ethical will, a
 * letter of values left to one's family. They are offered when asked for, never pushed.
 */
export const LEGACY_PROMPTS: readonly { id: string; text: string }[] = [
  { id: "life-story", text: "Tell the story of your life: the parts you remember most, or think matter most." },
  { id: "proud", text: "What are you most proud of?" },
  { id: "known", text: "What do you want the people you love, and the world, to know about you?" },
  { id: "learned", text: "What have you learned about life that you would want to pass on?" },
  { id: "hopes", text: "What are your hopes for the people you love?" },
  { id: "remembered", text: "How do you want the world to remember you?" },
  { id: "values", text: "What did you live by? Name the values you hope outlast you, and where they came from." },
  { id: "forgive", text: "Is there anything you want to ask forgiveness for, or to forgive?" },
  { id: "unsaid", text: "What have you never said to someone that you want them to hear?" },
  { id: "funeral", text: "What would you want said at your funeral, and by whom?" },
];

export const legacyPrompt = (id: string) => LEGACY_PROMPTS.find((p) => p.id === id);

/** Problems with life/ and needs/ for `reflect check`: a state.json that does not match its passages. */
export async function lifetimeProblems(root: string, paths: RecordPaths): Promise<string[]> {
  if (paths.person) return [];
  const out: string[] = [];
  const passagesText = await readRecord(root, join("life", "passages.jsonl"));
  const stateText = await readRecord(root, join("life", "state.json"));
  let passages: Passage[] = [];
  try {
    passages = parsePassages(passagesText);
  } catch {
    out.push("life/passages.jsonl is not JSON lines");
  }
  for (const p of passages) if (!isRegion(p.region) || !(REGIONS[p.region] as readonly string[]).includes(p.to)) out.push(`life/passages.jsonl: ${p.region} has no state ${p.to}`);
  if (stateText !== null && JSON.stringify(JSON.parse(stateText)) !== JSON.stringify(lifeState(passages))) out.push("life/state.json does not match life/passages.jsonl; it is rebuilt from the passages, never edited");
  return out;
}
