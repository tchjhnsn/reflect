// Items and their pipeline (docs/design/REFLECT_DOMAIN_2026-11-12.md, "The pipeline" and "Item states";
// REFLECT_SYSTEM_2026-11-12.md section 3). An item is something the person has stated: a belief, a plan, a
// need, something they noticed in what they did. It moves Observe, Reflect, Examine, Plan, Act. Every change
// is an event, dated and never edited; the item's state is a fold over its events, and an event the item
// statechart does not allow is refused with the reason. This file decides; it has no database and no network.
import type { Act } from "./log";

export type ItemState = "observed" | "reflected" | "examining" | "planned" | "scheduled" | "in_progress" | "done" | "missed" | "dropped";

/** Reflect's outcomes: now that it is stated, do you still hold it? */
export const REFLECT_OUTCOMES = ["held", "revised", "unsure", "dropped"] as const;
export type ReflectOutcome = (typeof REFLECT_OUTCOMES)[number];

/** Examine's steps (proposed in the domain design): state it precisely (Naess), ground it, set it against things (Rawls), take counsel. */
export const EXAMINE_STEPS = [
  { id: "precise", label: "State it precisely", ask: "Say exactly what you mean. Which words could be read more than one way?" },
  { id: "ground", label: "Ground it", ask: "Why do you hold it? Ask why again until you reach something you hold for its own sake." },
  { id: "against", label: "Set it against things", ask: "What is the strongest case against it? What does the evidence say? Does it fit what else you hold?" },
  { id: "counsel", label: "Take counsel", ask: "Who did you talk it over with, and what did they say?" },
] as const;
export type ExamineStep = (typeof EXAMINE_STEPS)[number]["id"];

/** Examine's outcomes, on two axes that change independently (decided). After AGM belief revision and Toulmin. */
export const BELIEVED = ["kept", "qualified", "revised", "withdrawn", "suspended"] as const;
export type Believed = (typeof BELIEVED)[number];
export const STANDS = ["grounded", "reinforced", "restated", "unchanged"] as const;
export type Stands = (typeof STANDS)[number];

/** What an item was stated from: an entry in the journal, or an act in the log (kept as it read then). */
export type ItemSource = { kind: "entry"; id: string } | { kind: "act"; act: Act };

export type ItemMove =
  | { kind: "stated"; text: string; from?: ItemSource }
  | { kind: "observed_why"; text: string }
  | { kind: "reflected"; outcome: ReflectOutcome; text?: string }
  | { kind: "examine" }
  | { kind: "examine_step"; step: ExamineStep; text: string }
  | { kind: "examined"; believed: Believed; stands: Stands; text?: string }
  | { kind: "another_look" }
  | { kind: "planned"; text: string }
  | { kind: "scheduled"; on: string }
  | { kind: "started" }
  | { kind: "done"; text?: string }
  | { kind: "missed"; text?: string };
export type ItemEvent = ItemMove & { at: string };

export type Item = {
  state: ItemState;
  /** The statement as it stands now: the last wording stated, revised, qualified or restated. */
  statement: string;
  from: ItemSource | null;
  plan: string | null;
  scheduledOn: string | null;
  /** The last outcome of Reflect and of Examine. */
  reflected: ReflectOutcome | null;
  examined: { believed: Believed; stands: Stands } | null;
  /** Examine's notes while examining, by step, in order. */
  notes: { step: ExamineStep; text: string; at: string }[];
  events: ItemEvent[];
};

/** Which moves each state allows: the item statechart. A first `stated` makes the item. */
const ALLOWED: Record<ItemState, ItemMove["kind"][]> = {
  observed: ["reflected"],
  reflected: ["examine", "planned", "reflected"],
  examining: ["examine_step", "examined", "another_look", "planned"],
  planned: ["scheduled", "examine"],
  scheduled: ["started", "scheduled"],
  in_progress: ["done", "missed", "examine"],
  done: ["examine"],
  missed: ["observed_why"],
  dropped: [],
};

export const allowedMoves = (state: ItemState): ItemMove["kind"][] => ALLOWED[state];

const words = (t: unknown, max = 4000) => (typeof t === "string" ? t.replace(/\r\n?/g, "\n").trim().slice(0, max) : "");
const isDay = (d: unknown): d is string => typeof d === "string" && /^\d{4}-\d{2}-\d{2}$/.test(d) && !Number.isNaN(Date.parse(d));

/**
 * Why `move` cannot be made on `item` (null for a new item), or null when it can. The statechart, plus what
 * each move needs: words where words are the point, a new wording for a revision, a day to schedule.
 */
export function refuse(item: Item | null, move: ItemMove): string | null {
  if (!item) return move.kind === "stated" ? (words(move.text) ? null : "Say what you observe first.") : "Nothing is stated yet.";
  if (!ALLOWED[item.state].includes(move.kind)) {
    if (item.state === "dropped") return "This item was dropped; it stays in the record and leaves the pipeline.";
    if (item.state === "missed") return "It was missed: observe why first, then go round again.";
    return `An item that is ${item.state.replace("_", " ")} cannot be ${move.kind.replace("_", " ")}.`;
  }
  switch (move.kind) {
    case "observed_why":
      return words(move.text) ? null : "Say why it was missed.";
    case "reflected":
      if (!REFLECT_OUTCOMES.includes(move.outcome)) return "That is not an outcome of Reflect.";
      if (item.state === "reflected" && move.outcome !== "dropped") return "It is already reflected on; it can still be dropped.";
      return move.outcome === "revised" && !words(move.text) ? "Write how you hold it now." : null;
    case "examine_step":
      if (!EXAMINE_STEPS.some((s) => s.id === move.step)) return "That is not a step of Examine.";
      return words(move.text) ? null : "Write something for this step.";
    case "examined":
      if (!BELIEVED.includes(move.believed) || !STANDS.includes(move.stands)) return "That is not an outcome of Examine.";
      return (move.believed === "revised" || move.believed === "qualified" || move.stands === "restated") && !words(move.text) ? "Write the new wording." : null;
    case "planned":
      return words(move.text) ? null : "Say how you will carry it out.";
    case "scheduled":
      return isDay(move.on) ? null : "Pick a day.";
    default:
      return null;
  }
}

/** The next state after a move that `refuse` allowed. */
function next(move: ItemMove): ItemState {
  switch (move.kind) {
    case "stated": case "observed_why": return "observed";
    case "reflected": return move.outcome === "dropped" ? "dropped" : "reflected";
    case "examine": return "examining";
    case "examine_step": return "examining";
    case "examined": return move.believed === "withdrawn" ? "dropped" : "examining";
    case "another_look": return "reflected";
    case "planned": return "planned";
    case "scheduled": return "scheduled";
    case "started": return "in_progress";
    case "done": return "done";
    case "missed": return "missed";
  }
}

/** Apply one event. Throws with the reason when the statechart refuses it. */
export function apply(item: Item | null, e: ItemEvent): Item {
  const why = refuse(item, e);
  if (why) throw new Error(why);
  const base: Item = item ?? { state: "observed", statement: "", from: null, plan: null, scheduledOn: null, reflected: null, examined: null, notes: [], events: [] };
  const out: Item = { ...base, state: next(e), events: [...base.events, e] };
  switch (e.kind) {
    case "stated":
      out.statement = words(e.text);
      out.from = e.from ?? null;
      break;
    case "observed_why":
      // Observing why it was missed: the item goes round again with what was learned; the statement stands.
      out.reflected = null;
      out.examined = null;
      out.notes = [];
      break;
    case "reflected":
      out.reflected = e.outcome;
      if (e.outcome === "revised") out.statement = words(e.text);
      break;
    case "examine":
      out.notes = [];
      out.examined = null;
      break;
    case "examine_step":
      out.notes = [...base.notes, { step: e.step, text: words(e.text), at: e.at }];
      break;
    case "examined":
      out.examined = { believed: e.believed, stands: e.stands };
      if (words(e.text)) out.statement = words(e.text);
      break;
    case "planned":
      out.plan = words(e.text);
      break;
    case "scheduled":
      out.scheduledOn = e.on;
      break;
  }
  return out;
}

/** An item from its events, oldest first. Throws if the events break the statechart. */
export function fold(events: readonly ItemEvent[]): Item | null {
  return events.reduce<Item | null>((it, e) => apply(it, e), null);
}

/** The phase of the pipeline an item sits in, for showing where it is. */
export type Phase = "Observe" | "Reflect" | "Examine" | "Plan" | "Act";
export const PHASES: Phase[] = ["Observe", "Reflect", "Examine", "Plan", "Act"];
export function phaseOf(state: ItemState): Phase | null {
  switch (state) {
    case "observed": case "missed": return "Observe";
    case "reflected": return "Reflect";
    case "examining": return "Examine";
    case "planned": return "Plan";
    case "scheduled": case "in_progress": case "done": return "Act";
    case "dropped": return null;
  }
}

/** How a state reads to the person. */
export const STATE_LABEL: Record<ItemState, string> = {
  observed: "Observed", reflected: "Reflected", examining: "Under examination", planned: "Planned",
  scheduled: "Scheduled", in_progress: "In progress", done: "Done", missed: "Missed", dropped: "Dropped",
};

/** How an event reads in the item's history. */
export function eventText(e: ItemEvent): string {
  switch (e.kind) {
    case "stated": return `Stated: ${e.text}`;
    case "observed_why": return `Observed why it was missed: ${e.text}`;
    case "reflected": return e.outcome === "revised" ? `Reflected: revised to "${e.text}"` : `Reflected: ${e.outcome}`;
    case "examine": return "Put under examination";
    case "examine_step": return `${EXAMINE_STEPS.find((s) => s.id === e.step)?.label ?? e.step}: ${e.text}`;
    case "examined": return `Examined: ${e.believed}, ${e.stands}${e.text ? `. Now: "${e.text}"` : ""}`;
    case "another_look": return "Needs another look";
    case "planned": return `Planned: ${e.text}`;
    case "scheduled": return `Scheduled for ${e.on}`;
    case "started": return "Started";
    case "done": return e.text ? `Done: ${e.text}` : "Done";
    case "missed": return e.text ? `Missed: ${e.text}` : "Missed";
  }
}

/** A move as the person sent it, checked field by field (it comes from a form). */
export function readMove(x: unknown): ItemMove | null {
  if (!x || typeof x !== "object") return null;
  const m = x as Record<string, unknown>;
  const text = typeof m.text === "string" ? m.text : undefined;
  switch (m.kind) {
    case "stated": return { kind: "stated", text: text ?? "" };
    case "observed_why": return { kind: "observed_why", text: text ?? "" };
    case "reflected": return REFLECT_OUTCOMES.includes(m.outcome as ReflectOutcome) ? { kind: "reflected", outcome: m.outcome as ReflectOutcome, text } : null;
    case "examine": return { kind: "examine" };
    case "examine_step": return EXAMINE_STEPS.some((s) => s.id === m.step) ? { kind: "examine_step", step: m.step as ExamineStep, text: text ?? "" } : null;
    case "examined": return BELIEVED.includes(m.believed as Believed) && STANDS.includes(m.stands as Stands) ? { kind: "examined", believed: m.believed as Believed, stands: m.stands as Stands, text } : null;
    case "another_look": return { kind: "another_look" };
    case "planned": return { kind: "planned", text: text ?? "" };
    case "scheduled": return typeof m.on === "string" ? { kind: "scheduled", on: m.on } : null;
    case "started": return { kind: "started" };
    case "done": return { kind: "done", text };
    case "missed": return { kind: "missed", text };
    default: return null;
  }
}
