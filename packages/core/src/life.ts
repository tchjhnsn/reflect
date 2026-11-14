// The apps behind the home screen's other tiles (Torian, 2026-11-14 15:45Z, "Let's build them"), from the domain
// design (docs/design/REFLECT_DOMAIN_2026-11-12.md, "Subdomains, one at a time") and the interface design (section
// 3: which tiles open which app). Each app keeps the person's records of a few kinds, each record with a state
// the person sets and acts they keep (a workout, a vote, a time you saw a friend). Everything is stated by the
// person, never inferred; the journal is where they reflect on it. Pure data and arithmetic: no I/O.
import type { Act } from "./log";

export const LIFE_APPS = ["work", "home", "body", "friends", "mate", "kin", "polis", "legacy"] as const;
export type LifeAppId = (typeof LIFE_APPS)[number];
export const isLifeApp = (s: unknown): s is LifeAppId => typeof s === "string" && (LIFE_APPS as readonly string[]).includes(s);

export type Option = { id: string; label: string };
export type FieldKind = "text" | "long" | "number" | "money" | "day" | "choice" | "check";
export type Field = { key: string; label: string; kind: FieldKind; options?: readonly Option[]; hint?: string };
export type State = { id: string; label: string; /** A record in this state is under way, which makes its tile "in action". */ active?: boolean };
export type ActKind = { id: string; label: string; /** Past tense for the log: "Worked out". */ done: string; minutes?: boolean; amount?: boolean };
export type Kind = {
  id: string;
  /** One record ("Friend") and many ("Friends"). */
  label: string;
  plural: string;
  /** The field shown as the record's name. */
  title: string;
  fields: readonly Field[];
  states?: readonly State[];
  acts?: readonly ActKind[];
  /** The person keeps one of these (their intended household, how food is met), not a list. */
  one?: boolean;
  /** What the kind is for, in a sentence, shown above it. */
  intro: string;
  /** A standard to hold it against, with its source, shown beside it. */
  standard?: { text: string; source: string };
};
export type LifeApp = { id: LifeAppId; label: string; intro: string; kinds: readonly Kind[] };

const opts = (...l: string[]): Option[] => l.map((x) => ({ id: x.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""), label: x }));

/** How a need is met and by whom (adopted 2026-11-09; gifts added in the domain design's Income section). */
export const MET = opts("Self-provided", "Stocked", "Supplied", "Cared for");
export const PROVIDER = opts("Self", "Market", "Nature", "Polis", "Kin", "Gifts");

const need = (id: string, label: string, intro: string, standard: { text: string; source: string }, checks: [string, string][]): Kind => ({
  id, label, plural: label, title: "how", one: true, intro, standard,
  fields: [
    { key: "how", label: "How it is met", kind: "choice", options: MET },
    { key: "by", label: "By whom", kind: "choice", options: PROVIDER },
    ...checks.map(([key, label]) => ({ key, label, kind: "check" as const })),
    { key: "cost", label: "What it costs a month", kind: "money" },
    { key: "note", label: "Notes", kind: "long" },
  ],
});

export const APPS: readonly LifeApp[] = [
  {
    id: "work", label: "Work",
    intro: "Work and provide. Unpaid work counts as work, and paid work feeds your income.",
    kinds: [
      {
        id: "work", label: "Work", plural: "Work", title: "role",
        intro: "Each piece of work you do, paid or unpaid: a job, a venture, keeping a home.",
        fields: [
          { key: "role", label: "Role", kind: "text", hint: "Electrician, founder, raising our children" },
          { key: "where", label: "Where or for whom", kind: "text" },
          { key: "paid", label: "Paid", kind: "choice", options: opts("Paid", "Unpaid") },
          { key: "hours", label: "Hours a week", kind: "number" },
          { key: "since", label: "Since", kind: "day" },
          { key: "keep", label: "Would you keep doing it if your needs were met another way?", kind: "choice", options: opts("Yes", "No", "Not sure") },
          { key: "note", label: "Notes", kind: "long" },
        ],
        states: [{ id: "seeking", label: "Seeking" }, { id: "working", label: "Working", active: true }, { id: "between", label: "Between work" }, { id: "ended", label: "Ended" }],
        acts: [{ id: "worked", label: "Hours worked", done: "Worked", minutes: true }],
      },
      {
        id: "income", label: "Income", plural: "Income", title: "source",
        intro: "Where your money comes from, how much and how regularly. Gifts and donations count.",
        fields: [
          { key: "source", label: "Source", kind: "text", hint: "Wages from the shop, a gift from church" },
          { key: "amount", label: "Amount", kind: "money" },
          { key: "per", label: "Per", kind: "choice", options: opts("Hour", "Week", "Month", "Year", "Once") },
          { key: "hours", label: "Hours a week, when paid by the hour", kind: "number" },
          { key: "regular", label: "How regular", kind: "choice", options: opts("Steady", "Varies", "Once") },
          { key: "note", label: "Notes", kind: "long" },
        ],
        states: [{ id: "coming-in", label: "Coming in", active: true }, { id: "expected", label: "Expected" }, { id: "ended", label: "Ended" }],
        acts: [{ id: "paid", label: "Paid", done: "Was paid", amount: true }],
      },
      {
        id: "household", label: "Intended household", plural: "Intended household", title: "where", one: true,
        intro: "The household you are heading toward, which your income is measured against.",
        standard: { text: "A livable income for this household where you live, from MIT's Living Wage Calculator (livingwage.mit.edu), which you look up and enter here.", source: "MIT Living Wage Calculator" },
        fields: [
          { key: "adults", label: "Adults", kind: "number" },
          { key: "working", label: "Of them, working", kind: "number" },
          { key: "children", label: "Children", kind: "number" },
          { key: "where", label: "Where", kind: "text", hint: "County and state" },
          { key: "when", label: "By when", kind: "day" },
          { key: "need", label: "What it needs a year before taxes", kind: "money" },
          { key: "note", label: "Notes", kind: "long" },
        ],
      },
      {
        id: "credential", label: "Credential", plural: "Credentials", title: "name",
        intro: "Licenses, degrees and certifications a goal requires, and where you are with each.",
        fields: [
          { key: "name", label: "Credential", kind: "text", hint: "Journeyman electrician license" },
          { key: "for", label: "Needed for", kind: "text" },
          { key: "from", label: "Granted by", kind: "text" },
          { key: "note", label: "Notes", kind: "long" },
        ],
        states: [{ id: "needed", label: "Needed" }, { id: "working-on", label: "Working on it", active: true }, { id: "held", label: "Held" }, { id: "lapsed", label: "Lapsed" }],
      },
    ],
  },
  {
    id: "home", label: "Home",
    intro: "The needs of self-preservation: how each is met now, by whom, and against what standard.",
    kinds: [
      need("food", "Food", "How you eat: who cooks, what is stocked, what it costs.", { text: "USDA food plans set what a household's food costs, from thrifty to liberal; a stocked pantry for emergencies.", source: "USDA Food Plans" }, [["pantry", "A pantry stocked for two weeks"]]),
      need("water", "Water", "Where your water comes from and what is stored.", { text: "Store a gallon per person per day, for at least three days and up to two weeks.", source: "Ready.gov" }, [["stored", "Water stored for three days"], ["two-weeks", "Water stored for two weeks"]]),
      need("shelter", "Shelter", "Where you live: rented or owned, and what it costs.", { text: "Housing costs under 30% of income.", source: "HUD" }, [["thirty", "Housing under 30% of income"]]),
      need("transport", "Transport", "How you get to work and essentials, and what it costs.", { text: "Reliable access to work and essentials at a cost you can keep up.", source: "Domain design" }, [["reliable", "Reliable every day"], ["backup", "A second way when the first fails"]]),
      need("safety", "Safety", "Physical safety at home and away, an emergency fund and preparedness.", { text: "An emergency kit and plan, an emergency fund, and important documents kept safe.", source: "Ready.gov" }, [["kit", "An emergency kit"], ["plan", "A plan the household knows"], ["fund", "An emergency fund"], ["documents", "Important documents kept safe"]]),
      {
        id: "firearm", label: "Firearm", plural: "Firearms", title: "name",
        intro: "Each firearm you own: training, safe storage (especially with children in the home) and permits, which state law sets.",
        fields: [
          { key: "name", label: "Firearm", kind: "text" },
          { key: "training", label: "Trained on it", kind: "check" },
          { key: "storage", label: "Stored safely", kind: "check" },
          { key: "permit", label: "Permit, where required", kind: "text" },
          { key: "note", label: "Notes", kind: "long" },
        ],
        acts: [{ id: "trained", label: "Training", done: "Trained with", minutes: true }],
      },
    ],
  },
  {
    id: "body", label: "Body",
    intro: "Health, with rest and sleep, and fitness, its own part.",
    kinds: [
      {
        id: "health", label: "Health", plural: "Health", title: "doctor", one: true,
        intro: "Coverage, a regular doctor, and how you rest and sleep.",
        standard: { text: "Health coverage, a regular doctor, and seven or more hours of sleep a night for adults.", source: "CDC" },
        fields: [
          { key: "coverage", label: "Coverage", kind: "text" },
          { key: "doctor", label: "Regular doctor", kind: "text" },
          { key: "checkup", label: "Last checkup", kind: "day" },
          { key: "sleep", label: "Hours of sleep a night, usually", kind: "number" },
          { key: "note", label: "Notes", kind: "long" },
        ],
        acts: [{ id: "appointment", label: "Appointment", done: "Went to an appointment" }, { id: "slept", label: "Sleep", done: "Slept", minutes: true }],
      },
      {
        id: "fitness", label: "Fitness goal", plural: "Fitness", title: "goal",
        intro: "Your own goals, beyond the floor of 150 minutes of moderate activity and two days of strength a week.",
        standard: { text: "150 minutes a week of moderate activity (or 75 vigorous) and muscle strengthening on two days.", source: "Physical Activity Guidelines for Americans" },
        fields: [
          { key: "goal", label: "Goal", kind: "text", hint: "Run a 5K, squat my bodyweight" },
          { key: "by", label: "By when", kind: "day" },
          { key: "note", label: "Notes", kind: "long" },
        ],
        states: [{ id: "set", label: "Set" }, { id: "in-progress", label: "In progress", active: true }, { id: "reached", label: "Reached" }, { id: "lapsed", label: "Lapsed" }],
        acts: [
          { id: "moderate", label: "Moderate activity", done: "Moderate activity toward", minutes: true },
          { id: "vigorous", label: "Vigorous activity", done: "Vigorous activity toward", minutes: true },
          { id: "strength", label: "Strength", done: "Strength training toward", minutes: true },
        ],
      },
    ],
  },
  {
    id: "friends", label: "Friends",
    intro: "Your friends and the groups you belong to. Notes about friends hold other people's lives: private, never published.",
    kinds: [
      {
        id: "friend", label: "Friend", plural: "Friends", title: "name",
        intro: "Who they are to you, how you met, and why you call them a friend.",
        fields: [
          { key: "name", label: "Name", kind: "text" },
          { key: "met", label: "How you met", kind: "long" },
          { key: "why", label: "Why you consider them a friend", kind: "long" },
          { key: "habits", label: "Habits you share", kind: "long" },
          { key: "kind", label: "Kind of friendship, if you use Aristotle's (Nicomachean Ethics VIII)", kind: "choice", options: opts("Utility", "Pleasure", "Virtue") },
        ],
        states: [{ id: "acquaintance", label: "Acquaintance" }, { id: "friend", label: "Friend", active: true }, { id: "close", label: "Close friend", active: true }, { id: "drifted", label: "Drifted" }],
        acts: [{ id: "saw", label: "Time together", done: "Spent time with" }, { id: "called", label: "Called or wrote", done: "Called or wrote" }],
      },
      {
        id: "group", label: "Group", plural: "Groups", title: "name",
        intro: "Teams, clubs, coalitions, a congregation.",
        fields: [{ key: "name", label: "Name", kind: "text" }, { key: "what", label: "What it is", kind: "text" }, { key: "note", label: "Notes", kind: "long" }],
        states: [{ id: "visiting", label: "Visiting" }, { id: "member", label: "Member", active: true }, { id: "serving", label: "Serving", active: true }, { id: "leading", label: "Leading", active: true }, { id: "left", label: "Left" }],
        acts: [{ id: "attended", label: "Attended", done: "Attended" }],
      },
    ],
  },
  {
    id: "mate", label: "Mate",
    intro: "Your own side of seeking a mate: what you look for, and each date and what you learned. Private.",
    kinds: [
      {
        id: "seeking", label: "What I am looking for", plural: "What I am looking for", title: "hopes", one: true,
        intro: "Must-haves, dealbreakers and hopes; your ideals about marriage and household; the household you intend.",
        fields: [
          { key: "must", label: "Must-haves", kind: "long" },
          { key: "never", label: "Dealbreakers", kind: "long" },
          { key: "hopes", label: "Hopes", kind: "long" },
          { key: "ideals", label: "Ideals about marriage and household", kind: "long" },
          { key: "household", label: "The household I intend", kind: "long" },
          { key: "where", label: "Where I meet people", kind: "long", hint: "Church, introductions, events, groups, apps" },
        ],
        states: [{ id: "seeking", label: "Seeking", active: true }, { id: "paused", label: "Not now" }, { id: "pair-bond", label: "In a pair bond" }],
      },
      {
        id: "date", label: "Date", plural: "Dates", title: "who",
        intro: "Each date: who, where, how it went and what you learned.",
        fields: [
          { key: "who", label: "Who", kind: "text" },
          { key: "day", label: "When", kind: "day" },
          { key: "where", label: "Where", kind: "text" },
          { key: "how", label: "How it went", kind: "long" },
          { key: "learned", label: "What I learned", kind: "long" },
          { key: "again", label: "See them again", kind: "choice", options: opts("Yes", "No", "Not sure") },
        ],
        states: [{ id: "planned", label: "Planned", active: true }, { id: "went", label: "Went" }, { id: "missed", label: "Missed" }],
      },
    ],
  },
  {
    id: "kin", label: "Kin",
    intro: "Kin you may care for one day, written down before any need starts, so the plan is made ahead and not in a crisis.",
    kinds: [
      {
        id: "relative", label: "Relative", plural: "Kin", title: "name",
        intro: "Who they are, their health, what they would want, and who would care for them.",
        fields: [
          { key: "name", label: "Name", kind: "text" },
          { key: "relation", label: "Relation", kind: "text", hint: "Mother, brother, grandfather" },
          { key: "born", label: "Born", kind: "day" },
          { key: "health", label: "Health", kind: "long" },
          { key: "wishes", label: "What they would want", kind: "long" },
          { key: "carer", label: "Who provides care", kind: "choice", options: opts("Me", "Family", "Paid help", "Given by church or charity", "No one yet") },
          { key: "note", label: "Notes", kind: "long" },
        ],
        states: [{ id: "independent", label: "Independent" }, { id: "some-help", label: "Needs some help", active: true }, { id: "regular-care", label: "Needs regular care", active: true }, { id: "full-care", label: "In full care", active: true }, { id: "died", label: "Died" }],
        acts: [{ id: "visited", label: "Visit", done: "Visited" }, { id: "cared", label: "Care given", done: "Cared for", minutes: true }],
      },
    ],
  },
  {
    id: "polis", label: "Polis",
    intro: "Civic life and your standing, all as you state it.",
    kinds: [
      {
        id: "civic", label: "Where I am a citizen", plural: "Where I am a citizen", title: "residence", one: true,
        intro: "Citizenship and residence, your districts and who represents you.",
        fields: [
          { key: "citizenship", label: "Citizenship", kind: "text" },
          { key: "residence", label: "Residence", kind: "text", hint: "City, county and state" },
          { key: "registered", label: "Registered to vote", kind: "check" },
          { key: "districts", label: "Districts", kind: "long" },
          { key: "representatives", label: "Representatives", kind: "long" },
          { key: "causes", label: "Questions I care about", kind: "long" },
        ],
        states: [{ id: "resident", label: "Resident" }, { id: "citizen", label: "Citizen" }, { id: "voting", label: "Voting", active: true }, { id: "serving", label: "Serving", active: true }, { id: "office", label: "Holding office", active: true }],
      },
      {
        id: "election", label: "Election", plural: "Elections", title: "name",
        intro: "Elections coming and past, and whether you voted.",
        fields: [{ key: "name", label: "Election", kind: "text" }, { key: "day", label: "Day", kind: "day" }, { key: "note", label: "What is on the ballot, and how I decided", kind: "long" }],
        states: [{ id: "coming", label: "Coming", active: true }, { id: "voted", label: "Voted" }, { id: "did-not-vote", label: "Did not vote" }],
      },
      {
        id: "standing", label: "Office or role", plural: "Standing", title: "role",
        intro: "Offices you hold, roles you are trusted with, boards and service. Aristotle's measure of standing is honor earned and deserved, not sought for itself.",
        fields: [{ key: "role", label: "Role", kind: "text" }, { key: "body", label: "Where", kind: "text" }, { key: "since", label: "Since", kind: "day" }, { key: "note", label: "Notes", kind: "long" }],
        states: [{ id: "serving", label: "Serving", active: true }, { id: "ended", label: "Ended" }],
        acts: [{ id: "served", label: "Service", done: "Served as", minutes: true }],
      },
      {
        id: "duty", label: "Duty", plural: "Duties", title: "name",
        intro: "Jury service, taxes, licenses and permits, and other things the law asks of you.",
        fields: [{ key: "name", label: "Duty", kind: "text" }, { key: "due", label: "Due", kind: "day" }, { key: "note", label: "Notes", kind: "long" }],
        states: [{ id: "due", label: "Due", active: true }, { id: "done", label: "Done" }],
      },
    ],
  },
  {
    id: "legacy", label: "Legacy",
    intro: "What you prepare for your death and what you want to survive you.",
    kinds: [
      {
        id: "plan", label: "Plan", plural: "Plans", title: "name",
        intro: "The will, life insurance, guardians for children, power of attorney, beneficiaries, end-of-life wishes. Revisit at each life event.",
        fields: [
          { key: "name", label: "What", kind: "choice", options: opts("Will", "Life insurance", "Guardians for children", "Power of attorney", "Beneficiaries", "End-of-life wishes", "Other") },
          { key: "who", label: "Who is named", kind: "text" },
          { key: "where", label: "Where it is kept", kind: "text" },
          { key: "reviewed", label: "Last reviewed", kind: "day" },
          { key: "note", label: "Notes", kind: "long" },
        ],
        states: [{ id: "unplanned", label: "Unplanned" }, { id: "planned", label: "Planned", active: true }, { id: "current", label: "Kept current" }],
        acts: [{ id: "reviewed", label: "Reviewed", done: "Reviewed" }],
      },
      {
        id: "vault", label: "What happens to my vaults", plural: "My vaults", title: "to", one: true,
        intro: "Named in advance: who receives your journal and records at your death, or that they are sealed.",
        fields: [{ key: "to", label: "Handed to", kind: "text", hint: "My children, my executor, or sealed" }, { key: "note", label: "What they should know", kind: "long" }],
      },
      {
        id: "wish", label: "Before I die", plural: "Bucket list", title: "what",
        intro: "What you want to do, see or make before you die.",
        fields: [{ key: "what", label: "What", kind: "text" }, { key: "note", label: "Notes", kind: "long" }],
        states: [{ id: "someday", label: "Someday" }, { id: "started", label: "Started", active: true }, { id: "done", label: "Done" }],
      },
      {
        id: "letter", label: "Letter", plural: "Letters", title: "to",
        intro: "A letter to someone, given to them after your death. Nothing here is released before then.",
        fields: [{ key: "to", label: "To", kind: "text" }, { key: "text", label: "Letter", kind: "long" }],
      },
    ],
  },
];

export function appOf(id: LifeAppId): LifeApp {
  return APPS.find((a) => a.id === id)!;
}
export function kindOf(app: LifeAppId, kind: string): Kind | null {
  return appOf(app).kinds.find((k) => k.id === kind) ?? null;
}

export type Body = Record<string, string | number | boolean>;

/** A record's fields from what the person sent, keeping only the kind's fields, each cleaned to its kind. */
export function cleanBody(k: Kind, raw: Record<string, unknown>): Body {
  const out: Body = {};
  for (const f of k.fields) {
    const v = raw[f.key];
    if (f.kind === "check") {
      if (v === true || v === "on" || v === "true") out[f.key] = true;
      continue;
    }
    if (v === undefined || v === null) continue;
    const s = String(v).trim();
    if (!s) continue;
    if (f.kind === "number" || f.kind === "money") {
      const n = Number(s.replace(/[$,\s]/g, ""));
      if (Number.isFinite(n) && n >= 0 && n < 1e10) out[f.key] = Math.round(n * 100) / 100;
    } else if (f.kind === "day") {
      if (/^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(Date.parse(s))) out[f.key] = s;
    } else if (f.kind === "choice") {
      if (f.options?.some((o) => o.id === s)) out[f.key] = s;
    } else out[f.key] = s.slice(0, f.kind === "long" ? 20000 : 300);
  }
  return out;
}

/** What a record is called: a kind kept once by the kind's name (Food), any other by its title field (a choice by its label). */
export function titleOf(k: Kind, b: Body): string {
  if (k.one) return k.label;
  const f = k.fields.find((x) => x.key === k.title);
  const v = b[k.title];
  if (v === undefined || v === "") return `Untitled ${k.label.toLowerCase()}`;
  if (f?.kind === "choice") return f.options?.find((o) => o.id === v)?.label ?? String(v);
  return String(v);
}

/** A field's value in words, or null when it is empty. */
export function shown(f: Field, b: Body): string | null {
  const v = b[f.key];
  if (v === undefined || v === "" || v === false) return null;
  if (f.kind === "check") return "Yes";
  if (f.kind === "choice") return f.options?.find((o) => o.id === v)?.label ?? String(v);
  if (f.kind === "money") return `$${Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
  return String(v);
}

export const isState = (k: Kind, s: unknown): s is string => typeof s === "string" && Boolean(k.states?.some((x) => x.id === s));
export const isAct = (k: Kind, s: unknown): s is string => typeof s === "string" && Boolean(k.acts?.some((x) => x.id === s));

/** What a tile opens: its app and, for the tiles that name one part of it, that kind. */
export const TILE_KIND: Record<string, { app: LifeAppId; kind?: string }> = {
  work: { app: "work", kind: "work" }, income: { app: "work", kind: "income" },
  food: { app: "home", kind: "food" }, water: { app: "home", kind: "water" }, shelter: { app: "home", kind: "shelter" }, transport: { app: "home", kind: "transport" }, safety: { app: "home", kind: "safety" },
  health: { app: "body", kind: "health" }, fitness: { app: "body", kind: "fitness" },
  friends: { app: "friends" }, standing: { app: "polis", kind: "standing" }, mate: { app: "mate" }, kin: { app: "kin" }, polis: { app: "polis" }, legacy: { app: "legacy" },
};

export type RecordFacts = { kind: string; state: string | null; body: Body };
export type ActFacts = { kind: string; record: string; day: string; minutes: number | null; amount: number | null };

/**
 * A tile's tone from the person's records in its app (or its one kind): in action when a record is in an active
 * state, a one-of-a-kind record is filled in and an act was kept in the last 30 days, or any act was kept in the
 * last 30 days; in thought when anything is stated; unstated otherwise.
 */
export function lifeTone(app: LifeAppId, kind: string | undefined, records: readonly (RecordFacts & { id: string })[], acts: readonly ActFacts[], today: string): "action" | "thought" | "unstated" {
  const mine = records.filter((r) => !kind || r.kind === kind);
  if (!mine.length) return "unstated";
  const ids = new Set(mine.map((r) => r.id));
  const since = new Date(Date.parse(`${today}T00:00:00Z`) - 30 * 86400_000).toISOString().slice(0, 10);
  if (acts.some((a) => ids.has(a.record) && a.day >= since)) return "action";
  const a = appOf(app);
  if (mine.some((r) => a.kinds.find((k) => k.id === r.kind)?.states?.find((s) => s.id === r.state)?.active)) return "action";
  return "thought";
}

const PER_YEAR: Record<string, number> = { hour: 52, week: 52, month: 12, year: 1, once: 0 };

/** A year of one income record, in dollars, or null when it cannot be said (no amount, or by the hour without hours). */
export function yearly(b: Body): number | null {
  const amount = typeof b.amount === "number" ? b.amount : null;
  if (amount === null) return null;
  const per = typeof b.per === "string" ? b.per : "month";
  if (per === "hour") return typeof b.hours === "number" ? amount * b.hours * 52 : null;
  return amount * (PER_YEAR[per] ?? 0);
}

/** Income coming in a year against what the intended household needs, from the person's own records. */
export function incomeSummary(records: readonly RecordFacts[]): { year: number; unknown: number; need: number | null; share: number | null } {
  let year = 0;
  let unknown = 0;
  for (const r of records) {
    if (r.kind !== "income" || (r.state && r.state !== "coming-in")) continue;
    const y = yearly(r.body);
    if (y === null) unknown++;
    else year += y;
  }
  const h = records.find((r) => r.kind === "household");
  const need = h && typeof h.body.need === "number" && h.body.need > 0 ? h.body.need : null;
  return { year: Math.round(year), unknown, need, share: need ? Math.round((year / need) * 100) : null };
}

/** Activity in the seven days ending today against the guidelines: vigorous minutes count twice. */
export function fitnessWeek(acts: readonly ActFacts[], today: string): { minutes: number; strengthDays: number; met: boolean } {
  const since = new Date(Date.parse(`${today}T00:00:00Z`) - 6 * 86400_000).toISOString().slice(0, 10);
  let minutes = 0;
  const strength = new Set<string>();
  for (const a of acts) {
    if (a.day < since || a.day > today) continue;
    if (a.kind === "moderate") minutes += a.minutes ?? 0;
    if (a.kind === "vigorous") minutes += 2 * (a.minutes ?? 0);
    if (a.kind === "strength") strength.add(a.day);
  }
  return { minutes, strengthDays: strength.size, met: minutes >= 150 && strength.size >= 2 };
}

/** How much of a need's standard is in place: its checks ticked out of all of them. */
export function checksOf(k: Kind, b: Body): { done: number; of: number } {
  const checks = k.fields.filter((f) => f.kind === "check");
  return { done: checks.filter((f) => b[f.key] === true).length, of: checks.length };
}

/**
 * The log's acts from the apps' records: each record kept, each state set after its first, and each act. A
 * record's first state is part of keeping it, not an act of its own. The log reads these, never copies them.
 */
export function lifeLogActs(records: readonly (RecordFacts & { id: string; app: LifeAppId; createdAt: string })[], states: readonly { record: string; state: string; at: string }[], acts: readonly ActFacts[]): Act[] {
  const out: Act[] = [];
  const byId = new Map(records.map((r) => [r.id, r]));
  for (const r of records) {
    const k = kindOf(r.app, r.kind);
    if (!k) continue;
    out.push({ at: r.createdAt, app: r.app, kind: "life_added", object: titleOf(k, r.body), detail: appOf(r.app).label });
  }
  const seen = new Set<string>();
  for (const s of [...states].sort((a, b) => a.at.localeCompare(b.at))) {
    const r = byId.get(s.record);
    const k = r && kindOf(r.app, r.kind);
    if (!r || !k) continue;
    if (!seen.has(s.record)) {
      seen.add(s.record);
      continue;
    }
    out.push({ at: s.at, app: r.app, kind: "life_state", object: titleOf(k, r.body), detail: k.states?.find((x) => x.id === s.state)?.label ?? s.state });
  }
  for (const a of acts) {
    const r = byId.get(a.record);
    const k = r && kindOf(r.app, r.kind);
    if (!r || !k) continue;
    const ak = k.acts?.find((x) => x.id === a.kind);
    const extra = a.minutes ? ` for ${a.minutes} minutes` : a.amount !== null ? ` $${a.amount.toLocaleString("en-US")}` : "";
    out.push({ at: a.day, app: r.app, kind: "life_act", object: `${titleOf(k, r.body)}${extra}`, detail: ak?.done ?? a.kind });
  }
  return out;
}
