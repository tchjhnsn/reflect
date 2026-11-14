// Browsing work (Torian, 2026-11-14 16:30Z): "browse jobs; those that exist, have existed, and are newly formed or
// forming or could be imagined to exist in the future ... browse companies and get inspired. It's not about income
// yet, it's about purpose and puzzle and reason." 16:33Z: "Fiction has imagined work, many people imagine work.
// We may curate." The catalogs are O*NET's occupations (jobs that exist, with the new tasks forming in them),
// Wikidata's historical professions (jobs that have existed), Y Combinator's directory (companies; the newest
// batch is forming), SEC filings (companies raising money or going public this week) and Wikidata's fictional
// professions and companies (work imagined in fiction), beside the person's own. Pure: no I/O.

export type Job = { code: string; title: string; about: string; zone?: number; tasks: string[]; emerging: string[]; interests: string[]; titles: string[]; related: string[]; /** BLS median pay a year, full time, and how many hold the job. */ pay?: number | null; people?: number | null };
export type WikiWork = { id: string; name: string; about: string; image: string | null; from: string | null; links: number; kind: "past" | "fiction-job" | "fiction-company" };
export type Company = { slug: string; name: string; line: string; about: string; batch: string; status: string; industries: string[]; logo: string | null; site: string | null; team: number | null; where: string | null; launched: number | null };
export type Filing = { name: string; form: "D" | "S-1"; day: string; where: string | null };

/** SOC 2018 major groups, the families O*NET's occupations belong to (by the first two digits of the code). */
export const FAMILIES: readonly { id: string; label: string }[] = [
  { id: "11", label: "Management" }, { id: "13", label: "Business and finance" }, { id: "15", label: "Computers and mathematics" },
  { id: "17", label: "Architecture and engineering" }, { id: "19", label: "Life, physical and social science" }, { id: "21", label: "Community and social service" },
  { id: "23", label: "Law" }, { id: "25", label: "Teaching and libraries" }, { id: "27", label: "Arts, design, entertainment, sports and media" },
  { id: "29", label: "Healthcare practice" }, { id: "31", label: "Healthcare support" }, { id: "33", label: "Protective service" },
  { id: "35", label: "Food preparation and serving" }, { id: "37", label: "Building and grounds" }, { id: "39", label: "Personal care and service" },
  { id: "41", label: "Sales" }, { id: "43", label: "Office and administrative support" }, { id: "45", label: "Farming, fishing and forestry" },
  { id: "47", label: "Construction and extraction" }, { id: "49", label: "Installation, maintenance and repair" }, { id: "51", label: "Production" },
  { id: "53", label: "Transportation and moving" }, { id: "55", label: "Military" },
];
export const familyOf = (code: string) => FAMILIES.find((f) => f.id === code.slice(0, 2)) ?? null;

/** O*NET's job zones: how much preparation the work takes. */
export const ZONES: Record<number, string> = { 1: "Little or no preparation", 2: "Some preparation", 3: "Medium preparation", 4: "Considerable preparation", 5: "Extensive preparation" };

/** Holland's six interests (RIASEC), as O*NET describes the work each favours. */
export const INTERESTS: Record<string, string> = {
  Realistic: "Hands-on work with things, tools, plants or animals",
  Investigative: "Ideas, searching for facts and figuring out problems",
  Artistic: "Forms, designs and patterns; self-expression",
  Social: "Working with, helping and teaching people",
  Enterprising: "Starting up and carrying out projects; leading and deciding",
  Conventional: "Following set procedures and routines; data and detail",
};

const norm = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();

/** Whether every word of a query is in the text, any order, ignoring case and accents. */
export function matches(q: string, ...text: (string | null | undefined)[]): boolean {
  const words = norm(q).split(" ").filter(Boolean);
  if (!words.length) return true;
  const hay = norm(text.filter(Boolean).join(" "));
  return words.every((w) => hay.includes(w));
}

/** A query's matches, best first: a name that starts with the query, then one that contains it, then the rest. */
export function ranked<T>(list: readonly T[], q: string, name: (t: T) => string, text: (t: T) => (string | null | undefined)[]): T[] {
  const n = norm(q);
  const hits = list.filter((t) => matches(q, name(t), ...text(t)));
  const score = (t: T) => (norm(name(t)).startsWith(n) ? 0 : norm(name(t)).includes(n) ? 1 : 2);
  return hits.map((t, i) => [t, score(t), i] as const).sort((a, b) => a[1] - b[1] || a[2] - b[2]).map(([t]) => t);
}

/** "Winter 2027" as a number that sorts batches in time (year, then the season's order). */
export function batchOrder(batch: string): number {
  const m = /^(Winter|Spring|Summer|Fall)\s+(\d{4})$/.exec(batch);
  if (!m) return 0;
  return Number(m[2]) * 10 + ["Winter", "Spring", "Summer", "Fall"].indexOf(m[1]!);
}

/** The batches forming now: the newest two in the directory. */
export function formingBatches(companies: readonly Company[]): string[] {
  return [...new Set(companies.map((c) => c.batch))].filter((b) => batchOrder(b) > 0).sort((a, b) => batchOrder(b) - batchOrder(a)).slice(0, 2);
}

/** Every industry in the directory, most companies first. */
export function industries(companies: readonly Company[]): { name: string; count: number }[] {
  const n = new Map<string, number>();
  for (const c of companies) for (const i of c.industries.slice(0, 1)) n.set(i, (n.get(i) ?? 0) + 1);
  return [...n].map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count);
}

/** Funds, trusts and vehicles file Form D too; these names are not companies being formed. */
const VEHICLE = /\b(fund|funds|lp|l\.p\.|trust|capital|partners|holdings|investors?|spv|co-?invest|series|portfolio|ventures? (fund|partners)|feeder|master|offshore|opportunit(y|ies)|real estate|reit|properties|property|dst|land|apartments?|residences|lofts|hospitality|hotels?|lending|mortgage|loans?|income|credit|equity|secondaries|aggregator|growth)\b/i;

/** EDGAR full-text search hits as filings of companies forming: funds and investment vehicles left out. */
export function filingsFrom(hits: readonly { _source?: { display_names?: string[]; form?: string; file_date?: string; biz_locations?: string[] } }[], form: "D" | "S-1"): Filing[] {
  const seen = new Set<string>();
  const out: Filing[] = [];
  for (const h of hits) {
    const raw = h._source?.display_names?.[0];
    if (!raw) continue;
    const name = raw.replace(/\s*\(CIK \d+\)\s*$/, "").replace(/\s+/g, " ").trim();
    if (form === "D" && VEHICLE.test(name)) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ name, form, day: h._source?.file_date ?? "", where: h._source?.biz_locations?.[0] ?? null });
  }
  return out;
}

/** What Work can point an inspiration at: one of the catalogs' entries, or one the person imagined. */
export type WorkRef = { kind: "job" | "past" | "company" | "fiction" | "mine"; id: string };
export function parseRef(s: string): WorkRef | null {
  const m = /^(job|past|company|fiction|mine):([A-Za-z0-9.\-_]{1,80})$/.exec(s);
  return m ? { kind: m[1] as WorkRef["kind"], id: m[2]! } : null;
}
export const refString = (r: WorkRef) => `${r.kind}:${r.id}`;
export function refHref(r: WorkRef): string {
  switch (r.kind) {
    case "job": return `/life/work/jobs/${r.id}`;
    case "past": return `/life/work/past/${r.id}`;
    case "company": return `/life/work/companies/${r.id}`;
    case "fiction": return `/life/work/imagined/${r.id}`;
    case "mine": return `/life/work/${r.id}`;
  }
}

/** n entries picked the same way all day and differently the next (a seeded shuffle), so browsing finds new work daily. */
export function daily<T>(list: readonly T[], n: number, day: string, salt = ""): T[] {
  let h = 2166136261;
  for (const ch of day + salt) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  const rand = () => ((h = Math.imul(h ^ (h >>> 15), 2246822507) ^ Math.imul(h ^ (h >>> 13), 3266489909)), (h >>> 0) / 4294967296);
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
  return a.slice(0, n);
}

// The person's workspace in Work (Torian, 2026-11-14 17:44Z): Learn and Work form a pipeline, and the other tiles
// inform Work with their context. Needs say what must be earned now, after what family and others cover; the time
// spent learning says what is left for work; what the person is learning and the work they want say which jobs
// build toward later. Every input is something the person recorded; nothing is inferred about them.

/** A need from Home as Work reads it: what it costs a month and who provides it. */
export type NeedFacts = { label: string; by: string | null; cost: number | null };
/** Who provides a need when the person does not pay for it themselves. */
const PROVIDED = new Set(["kin", "polis", "gifts", "nature"]);

/** What the person pays for their needs a year, what others cover, and which needs have no cost given. */
export function needsNow(needs: readonly NeedFacts[]): { pay: number | null; covered: { label: string; by: string; cost: number | null }[]; unknown: string[] } {
  let pay = 0;
  let known = false;
  const covered: { label: string; by: string; cost: number | null }[] = [];
  const unknown: string[] = [];
  for (const n of needs) {
    if (n.by && PROVIDED.has(n.by)) {
      covered.push({ label: n.label, by: n.by, cost: n.cost !== null ? n.cost * 12 : null });
      continue;
    }
    if (n.cost === null) {
      unknown.push(n.label);
      continue;
    }
    pay += n.cost * 12;
    known = true;
  }
  return { pay: known ? Math.round(pay) : null, covered, unknown };
}

/** What is left to earn a year: what the person needs, less what already comes in. Null when the need is not known. */
export const toEarn = (need: number | null, income: number) => (need === null ? null : Math.max(0, Math.round(need - income)));

/** A job's median pay a year for so many hours a week (BLS medians are for full time, 40 hours). */
export const payFor = (job: Pick<Job, "pay">, hours: number) => (job.pay ? Math.round((job.pay * Math.min(hours, 40)) / 40) : null);

const STOP = new Set(["and", "the", "for", "with", "from", "into", "about", "introduction", "intro", "principles", "basics", "fundamentals", "course", "lecture", "lectures", "part", "volume", "book", "theory", "history", "general", "advanced", "beginning", "learn", "learning", "guide", "workers", "work", "other", "all", "specialists", "science", "sciences", "studies", "systems", "natural", "social", "human", "applied", "special", "technicians", "assistants", "managers", "teachers", "postsecondary", "except"]);
/** The words of what the person is learning that can name work: four letters or more, cut to a stem of seven. */
export function stemsOf(texts: readonly string[]): string[] {
  const out = new Set<string>();
  for (const t of texts) for (const w of t.toLowerCase().normalize("NFKD").replace(/[^a-z ]+/g, " ").split(" ")) if (w.length >= 4 && !STOP.has(w)) out.add(w.slice(0, 7));
  return [...out];
}
const jobStems = (j: Job) => new Set(stemsOf([j.title, ...j.titles]));

/** A job suggested to the person, with why, in their terms. */
export type Suggestion = { job: Job; why: string[]; pay: number | null };

/**
 * Work for now: jobs that need little preparation (job zones 1 to 3), that would cover what is left to earn at the
 * hours the person can give, best first when they sit near work the person wants (O*NET's related jobs), use what
 * they are learning, or favour the same interests as the work they want.
 */
export function forNow(all: readonly Job[], p: { toEarn: number | null; hours: number | null; wanted: readonly Job[]; stems: readonly string[]; n?: number }): Suggestion[] {
  const hours = p.hours && p.hours > 0 ? p.hours : 40;
  const near = new Map<string, string>();
  for (const w of p.wanted) for (const r of w.related) if (!near.has(r)) near.set(r, w.title);
  const wantedCodes = new Set(p.wanted.map((w) => w.code));
  const interests = new Set(p.wanted.flatMap((w) => w.interests.slice(0, 1)));
  const stems = new Set(p.stems);
  const out: (Suggestion & { score: number })[] = [];
  for (const j of all) {
    if (wantedCodes.has(j.code) || !j.zone || j.zone > 3) continue;
    const pay = payFor(j, hours);
    if (p.toEarn && (pay === null || pay < p.toEarn)) continue;
    const why: string[] = [];
    let score = 0;
    const toward = near.get(j.code);
    if (toward) { why.push(`Near ${toward}, which you want`); score += 3; }
    const uses = [...jobStems(j)].filter((s) => stems.has(s));
    if (uses.length) { why.push("Uses what you are learning"); score += 2; }
    if (j.interests[0] && interests.has(j.interests[0])) { why.push(`${j.interests[0]} work, like what you want`); score += 1; }
    if (p.toEarn && pay) why.push(p.hours ? `Covers what is left to earn at ${hours} hours a week` : "Covers what is left to earn, full time");
    out.push({ job: j, why, pay, score: score * 10 - j.zone + (j.people ?? 0) / 1e7 });
  }
  // When some jobs connect to what the person wants or learns, show only those; otherwise the ones that cover the need.
  const linked = out.filter((s) => s.score >= 9);
  return (linked.length ? linked : out).sort((a, b) => b.score - a.score).slice(0, p.n ?? 8).map(({ score: _s, ...s }) => s);
}

/** Work for later: jobs that what the person is learning leads to, which they have not yet said they want. */
export function forLater(all: readonly Job[], p: { stems: readonly string[]; wanted: readonly Job[]; n?: number }): Suggestion[] {
  const stems = new Set(p.stems);
  if (!stems.size) return [];
  const wantedCodes = new Set(p.wanted.map((w) => w.code));
  const out: (Suggestion & { score: number })[] = [];
  for (const j of all) {
    if (wantedCodes.has(j.code)) continue;
    const hit = [...jobStems(j)].filter((s) => stems.has(s)).length;
    if (!hit) continue;
    out.push({ job: j, why: ["What you are learning leads here"], pay: j.pay ?? null, score: hit * 10 + (j.zone ?? 0) + (j.people ?? 0) / 1e7 });
  }
  // At most two from one family, so the work it leads to is not all teaching.
  const per = new Map<string, number>();
  const kept = out.sort((a, b) => b.score - a.score).filter((s) => {
    const f = s.job.code.slice(0, 2);
    per.set(f, (per.get(f) ?? 0) + 1);
    return per.get(f)! <= 2;
  });
  return kept.slice(0, p.n ?? 8).map(({ score: _s, ...s }) => s);
}
