// The log (docs/design/REFLECT_UI_2026-11-12.md section 8, 09:30Z): a dated record of what the person did in
// the tiles, kept by the apps themselves and never written by the person. It is not the journal (what they
// write and think) and not the chat history (what they said with Reflect). An act is something chosen and
// done (Aristotle's prohairesis); putting a book on the shelves is an intention and is shown as one.
// Acts are read from the apps' own records, never copied.
import type { BookStatus, Grasp } from "./learning";
import type { Stance } from "./claims";

export type ActKind = "intended" | "started" | "finished" | "put_down" | "cited" | "kept_claim" | "changed_stance" | "rated";
export type Act = {
  /** An ISO timestamp, or a date (YYYY-MM-DD) when the record keeps only the day. */
  at: string;
  app: "learn";
  kind: ActKind;
  /** What the act was on: a book's title, a claim's text, a subject's name. */
  object: string;
  /** The stance or grasp, for acts that have one. */
  detail?: string;
};

/** Learn's records as the log needs them, already opened. */
export type LearnFacts = {
  books: {
    title: string;
    addedAt: string;
    startedOn: string | null;
    finishedOn: string | null;
    /** The book's kept status changes (oldest first does not matter). Books shelved before they were kept have none. */
    statuses?: { status: BookStatus; day: string; at: string }[];
  }[];
  claims: { text: string; stances: { stance: Stance; at: string }[] }[];
  grasps: { subject: string; grasp: Grasp; at: string }[];
};

const STATUS_ACT: Record<BookStatus, ActKind> = { want: "intended", reading: "started", read: "finished", abandoned: "put_down", referenced: "cited" };
/** When a status change happened: the day the person gave if it is before the day it was kept, else the moment kept. */
const when = (s: { day: string; at: string }) => (s.day < s.at.slice(0, 10) ? s.day : s.at);

/**
 * Every act Learn's records show. A book with kept status changes gives one act per change; a book shelved
 * before they were kept gives its shelving and its start and finish days.
 */
export function learnActs(f: LearnFacts): Act[] {
  const acts: Act[] = [];
  for (const b of f.books) {
    if (b.statuses?.length) {
      for (const s of b.statuses) acts.push({ at: when(s), app: "learn", kind: STATUS_ACT[s.status], object: b.title });
      continue;
    }
    acts.push({ at: b.addedAt, app: "learn", kind: "intended", object: b.title });
    if (b.startedOn) acts.push({ at: b.startedOn, app: "learn", kind: "started", object: b.title });
    if (b.finishedOn) acts.push({ at: b.finishedOn, app: "learn", kind: "finished", object: b.title });
  }
  for (const c of f.claims) {
    [...c.stances].sort((a, b) => a.at.localeCompare(b.at)).forEach((s, i) => acts.push({ at: s.at, app: "learn", kind: i === 0 ? "kept_claim" : "changed_stance", object: c.text, detail: s.stance }));
  }
  for (const g of f.grasps) acts.push({ at: g.at, app: "learn", kind: "rated", object: g.subject, detail: g.grasp });
  return acts;
}

const STANCE: Record<string, string> = { belief: "a belief", disbelief: "a disbelief", suspension: "suspended judgment" };
const GRASP: Record<string, string> = { teach: "could teach", explain: "could explain the key ideas of", gist: "remember the gist of", forgot: "don't remember" };

/** One act in plain words. */
export function actText(a: Act): string {
  switch (a.kind) {
    case "intended": return `Put ${a.object} on your shelves`;
    case "started": return `Started reading ${a.object}`;
    case "finished": return `Finished ${a.object}`;
    case "put_down": return `Put down ${a.object}`;
    case "cited": return `Noted ${a.object} as a book you cite`;
    case "kept_claim": return `Kept a claim as ${STANCE[a.detail ?? ""] ?? a.detail}: ${a.object}`;
    case "changed_stance": return `Changed your stance to ${STANCE[a.detail ?? ""] ?? a.detail}: ${a.object}`;
    case "rated": return `Said you ${GRASP[a.detail ?? ""] ?? a.detail} ${a.object}`;
  }
}

/** The day an act falls on, in the given time zone offset in minutes (dates stay as they are). */
export const actDay = (a: Act, offsetMinutes = 0): string =>
  a.at.length === 10 ? a.at : new Date(Date.parse(a.at) + offsetMinutes * 60_000).toISOString().slice(0, 10);

/** Acts grouped by day, newest day first; within a day, timed acts newest first and day-only acts after them. */
export function logByDay(acts: readonly Act[], offsetMinutes = 0): { day: string; acts: Act[] }[] {
  const days = new Map<string, Act[]>();
  for (const a of acts) days.set(actDay(a, offsetMinutes), [...(days.get(actDay(a, offsetMinutes)) ?? []), a]);
  const order = (a: Act) => (a.at.length === 10 ? "" : a.at);
  return [...days.entries()]
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([day, list]) => ({ day, acts: list.sort((a, b) => order(b).localeCompare(order(a))) }));
}
