import { kinSpace } from "@reflect/core/tile-spaces";
import { incomeSummary, kindOf, type RecordFacts } from "@reflect/core/life";
import { forLater, forNow, needsNow, parseRef, stemsOf, toEarn, type Job, type Suggestion } from "@reflect/core/work-browse";
import { lifeActs, lifeRecords, type LifeAct, type LifeRecord } from "./life-store";
import { jobByCode, jobs } from "./work-catalog";
import { getWork } from "./work-store";
import { learningMinutes, listItems, type LearnItem } from "./ways-store";

// The person's Work workspace, read from what they recorded across their tiles (Torian, 2026-11-14 17:44Z): the work
// they want with what they wrote toward it, what Home's needs leave to earn, the time learning takes, and the jobs
// that fit now and that build toward later.

const NEEDS = ["food", "water", "shelter", "transport", "safety"] as const;
export const WANTED_KINDS = new Set(["inspiration", "imagined"]);

export type Wanted = { record: LifeRecord; title: string; job: Job | null; entries: number; lastEntry: string | null; learning: string[] };
export type Entry = { id: string; title: string; day: string; toward: string; towardId: string };
export type Workspace = {
  wanted: Wanted[];
  need: { pay: number | null; from: "home" | "household" | null; covered: { label: string; by: string; cost: number | null }[]; unknown: string[] };
  income: { year: number; unknown: number };
  left: number | null;
  hours: number | null;
  learningHours: number | null;
  careHours: number;
  items: LearnItem[];
  now: Suggestion[];
  later: Suggestion[];
  entries: Entry[];
};

const learningIds = (r: LifeRecord) => (typeof r.body.learning === "string" && r.body.learning ? r.body.learning.split(",") : []);
const titleOfWanted = (r: LifeRecord) => String(r.kind === "imagined" ? r.body.name ?? "" : r.body.title ?? "");

/** The jobs a wanted record points at: a job itself. */
export const jobOf = (r: LifeRecord): Job | null => {
  const ref = typeof r.body.ref === "string" ? parseRef(r.body.ref) : null;
  return ref?.kind === "job" ? jobByCode(ref.id) : null;
};

export async function workspace(userId: string, today = new Date().toISOString().slice(0, 10)): Promise<Workspace> {
  const [work, home, kin, acts, items] = await Promise.all([lifeRecords(userId, "work"), lifeRecords(userId, "home"), lifeRecords(userId, "kin"), lifeActs(userId), listItems(userId)]);
  const since = new Date(Date.parse(`${today}T00:00:00Z`) - 27 * 86400_000).toISOString().slice(0, 10);
  const minutes = await learningMinutes(userId, since);
  const wrote = acts.filter((a) => a.kind === "wrote");
  const byRecord = new Map<string, LifeAct[]>();
  for (const a of wrote) byRecord.set(a.record, [...(byRecord.get(a.record) ?? []), a]);

  const wantedRecords = work.filter((r) => WANTED_KINDS.has(r.kind) && r.state !== "let-go").reverse();
  const wanted: Wanted[] = wantedRecords.map((r) => {
    const mine = byRecord.get(r.id) ?? [];
    return { record: r, title: titleOfWanted(r), job: jobOf(r), entries: mine.length, lastEntry: mine[0]?.day ?? null, learning: learningIds(r) };
  });

  const n = needsNow(home.filter((r) => (NEEDS as readonly string[]).includes(r.kind)).map((r) => ({
    label: kindOf("home", r.kind)?.label ?? r.kind,
    by: typeof r.body.by === "string" ? r.body.by : null,
    cost: typeof r.body.cost === "number" ? r.body.cost : null,
  })));
  const facts: RecordFacts[] = work.map((r) => ({ kind: r.kind, state: r.state, body: r.body }));
  const inc = incomeSummary(facts);
  const pay = n.pay ?? inc.need;
  const from = n.pay !== null ? "home" : inc.need !== null ? "household" : null;
  const left = toEarn(pay, inc.year);
  const nowRec = work.find((r) => r.kind === "now");
  const hours = typeof nowRec?.body.hours === "number" ? nowRec.body.hours : null;

  const learnNow = items.filter((i) => i.status === "doing" || i.status === "want");
  const towardIds = new Set(wanted.flatMap((w) => w.learning));
  const stems = stemsOf([...learnNow, ...items.filter((i) => towardIds.has(i.id))].flatMap((i) => [i.title, ...i.subjects.map((s) => s.name)]));
  const wantedJobs = wanted.map((w) => w.job).filter((j): j is Job => j !== null);
  const all = jobs();

  const entries: Entry[] = [];
  for (const a of wrote.slice(0, 12)) {
    const w = wanted.find((x) => x.record.id === a.record) ?? null;
    const id = a.note ?? "";
    const doc = id ? await getWork(userId, id) : null;
    if (!doc) continue;
    entries.push({ id, title: doc.title || doc.body.replace(/^>.*\n+/, "").trim().split("\n")[0]?.slice(0, 80) || "Untitled", day: a.day, toward: w?.title ?? "", towardId: a.record });
  }

  return {
    wanted,
    need: { pay, from, covered: n.covered, unknown: n.unknown },
    income: { year: inc.year, unknown: inc.unknown },
    left,
    hours,
    learningHours: minutes ? Math.round((minutes / 60 / 4) * 10) / 10 : null,
    careHours: kinSpace(kin, acts, today).careHours,
    items,
    now: forNow(all, { toEarn: left, hours, wanted: wantedJobs, stems }),
    later: forLater(all, { stems, wanted: wantedJobs }),
    entries,
  };
}
