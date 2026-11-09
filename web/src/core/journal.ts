// The journal: the person's own words, with no model. Mirrors thrivesight_core/nts/journal.py.
import type { IsoDate } from "./person";

export const OPENING = "What would you like to remember?";
export const TELL_ME_MORE = "Tell me more.";
export const CLOSED = "Saved. You can leave it here, or reflect on it.";

export type Part = { seq: number; text: string; createdAt: string };

export type Entry = {
  id: string;
  localDate: IsoDate;
  createdAt: string;
  promptId: string | null;
  closedAt: string | null;
  parts: readonly Part[];
  reflections?: readonly Reflection[];
};

/** One run of reflecting on an entry: the frame picked and the answers given, in order. */
export type Reflection = {
  id: string;
  frame: string;
  createdAt: string;
  answers: readonly { seq: number; facet: string; stance: "descriptive" | "normative"; question: string; text: string }[];
};

/** What the journal shows next for an entry. Deterministic. */
export function nextPrompt(entry: Pick<Entry, "parts" | "closedAt">): string {
  if (entry.closedAt) return CLOSED;
  return entry.parts.length === 0 ? OPENING : TELL_ME_MORE;
}

export function clean(text: string): string | null {
  const t = text.replace(/\r\n/g, "\n").replace(/[ \t]+$/gm, "").trim();
  return t.length === 0 ? null : t;
}

/** Entries written on this month and day in earlier years, newest year first. */
export function onThisDay(entries: readonly Entry[], today: IsoDate): Entry[] {
  const md = today.slice(5);
  const year = today.slice(0, 4);
  return entries
    .filter((e) => e.localDate.slice(5) === md && e.localDate.slice(0, 4) < year)
    .sort((a, b) => b.localDate.localeCompare(a.localDate));
}

/** Group entries by the day they were written, newest day first, keeping each day's order newest first. */
export function byDay(entries: readonly Entry[]): { day: IsoDate; entries: Entry[] }[] {
  const days = new Map<IsoDate, Entry[]>();
  for (const e of [...entries].sort((a, b) => b.createdAt.localeCompare(a.createdAt))) {
    const list = days.get(e.localDate) ?? [];
    list.push(e);
    days.set(e.localDate, list);
  }
  return [...days.entries()].sort((a, b) => b[0].localeCompare(a[0])).map(([day, list]) => ({ day, entries: list }));
}

/** A local date is accepted when it is within a day of the server's date, so time zones work and back-dating does not. */
export function acceptLocalDate(local: IsoDate, serverNow: Date): boolean {
  const t = Date.parse(`${local}T12:00:00Z`);
  return Math.abs(t - serverNow.getTime()) <= 36 * 3600 * 1000;
}
