import { kindOf, type ActFacts, type RecordFacts } from "./life";

/** What the tiles' workspaces read: a person's records with their ids, and their acts. */
type Rec = RecordFacts & { id: string };

const DAY = 86400_000;
const at = (d: string) => Date.parse(`${d}T00:00:00Z`);
/** Days from one day to another, whole. */
export const daysBetween = (from: string, to: string) => Math.round((at(to) - at(from)) / DAY);
const back = (today: string, days: number) => new Date(at(today) - (days - 1) * DAY).toISOString().slice(0, 10);
const str = (v: unknown) => (typeof v === "string" && v ? v : null);
const stateLabel = (app: "friends" | "mate" | "kin" | "polis", r: Rec) => kindOf(app, r.kind)?.states?.find((s) => s.id === r.state)?.label ?? null;
const isActive = (app: "friends" | "mate" | "kin" | "polis", r: Rec) => Boolean(kindOf(app, r.kind)?.states?.find((s) => s.id === r.state)?.active);
const lastOf = (acts: readonly ActFacts[], id: string, kinds: readonly string[]) => acts.filter((a) => a.record === id && kinds.includes(a.kind)).map((a) => a.day).sort().at(-1) ?? null;

/**
 * Friends' workspace: friends and close friends, longest since time together (or a call) first, so who has gone
 * quiet comes up; then the rest; and the groups the person is in, with when they last went.
 */
export function friendsSpace(records: readonly Rec[], acts: readonly ActFacts[], today: string) {
  const friends = records.filter((r) => r.kind === "friend").map((r) => {
    const saw = lastOf(acts, r.id, ["saw"]);
    const called = lastOf(acts, r.id, ["called"]);
    const last = [saw, called].filter((d): d is string => d !== null).sort().at(-1) ?? null;
    return { id: r.id, name: str(r.body.name) ?? "A friend", state: stateLabel("friends", r), active: isActive("friends", r), saw, called, since: last ? daysBetween(last, today) : null };
  });
  const order = (a: { since: number | null }, b: { since: number | null }) => (b.since ?? Infinity) - (a.since ?? Infinity);
  const groups = records.filter((r) => r.kind === "group" && isActive("friends", r)).map((r) => ({ id: r.id, name: str(r.body.name) ?? "A group", state: stateLabel("friends", r), last: lastOf(acts, r.id, ["attended"]) }));
  return { close: friends.filter((f) => f.active).sort(order), others: friends.filter((f) => !f.active), groups };
}

/** Mate's workspace: whether the person is seeking and what they look for, dates planned (soonest first) and dates gone on (latest first). */
export function mateSpace(records: readonly Rec[], today: string) {
  const s = records.find((r) => r.kind === "seeking");
  const dates = records.filter((r) => r.kind === "date").map((r) => ({ id: r.id, who: str(r.body.who) ?? "Someone", day: str(r.body.day), where: str(r.body.where), state: r.state, learned: str(r.body.learned), again: str(r.body.again) }));
  return {
    seeking: s ? { id: s.id, state: stateLabel("mate", s), must: str(s.body.must), never: str(s.body.never), household: str(s.body.household) } : null,
    planned: dates.filter((d) => d.state === "planned" && (!d.day || d.day >= today)).sort((a, b) => (a.day ?? "9").localeCompare(b.day ?? "9")),
    went: dates.filter((d) => d.state === "went").sort((a, b) => (b.day ?? "").localeCompare(a.day ?? "")),
  };
}

/**
 * Kin's workspace: kin who need care first, each with who provides it, the care given and when last visited; the
 * rest with their age; and the hours a week of care the person gave over four weeks, which is time Work cannot use.
 */
export function kinSpace(records: readonly Rec[], acts: readonly ActFacts[], today: string) {
  const four = back(today, 28);
  const care = (id: string) => acts.filter((a) => a.kind === "cared" && a.record === id && a.day >= four && a.day <= today).reduce((s, a) => s + (a.minutes ?? 0), 0);
  const kin = records.filter((r) => r.kind === "relative" && r.state !== "died").map((r) => {
    const born = str(r.body.born);
    return { id: r.id, name: str(r.body.name) ?? "Kin", relation: str(r.body.relation), state: stateLabel("kin", r), needsCare: isActive("kin", r), carer: str(r.body.carer), age: born ? Math.floor(daysBetween(born, today) / 365.25) : null, careMinutes: care(r.id), visited: lastOf(acts, r.id, ["visited"]) };
  });
  const minutes = kin.reduce((s, k) => s + k.careMinutes, 0);
  return { care: kin.filter((k) => k.needsCare), others: kin.filter((k) => !k.needsCare), careHours: Math.round((minutes / 60 / 4) * 10) / 10 };
}

/** Polis' workspace: where the person is a citizen and whether registered; elections coming and duties due, soonest first; and the offices and roles they serve in. */
export function polisSpace(records: readonly Rec[], today: string) {
  const c = records.find((r) => r.kind === "civic");
  const dated = (kind: string, state: string, key: string) => records.filter((r) => r.kind === kind && r.state === state).map((r) => ({ id: r.id, name: str(r.body.name) ?? "", day: str(r.body[key]), late: Boolean(str(r.body[key]) && str(r.body[key])! < today) })).sort((a, b) => (a.day ?? "9").localeCompare(b.day ?? "9"));
  return {
    civic: c ? { id: c.id, state: stateLabel("polis", c), residence: str(c.body.residence), citizenship: str(c.body.citizenship), registered: c.body.registered === true } : null,
    elections: dated("election", "coming", "day"),
    duties: dated("duty", "due", "due"),
    serving: records.filter((r) => r.kind === "standing" && r.state === "serving").map((r) => ({ id: r.id, role: str(r.body.role) ?? "A role", where: str(r.body.body) })),
  };
}
