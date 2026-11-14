// The calendar, a tool (Torian, 2026-11-14 17:44Z: "Each tile will also need its own calendar (tags), and the user
// will need a master calendar"; 17:52Z: "a calendar is a tool"). An event is tagged with the tile it belongs to; a
// tile's calendar is the master calendar filtered to its tag. Repeats follow iCalendar's frequencies (RFC 5545
// FREQ=DAILY, WEEKLY, MONTHLY, YEARLY) with an optional end day. Pure: no I/O.

export const CAL_TILES = [
  { id: "learn", label: "Learn" }, { id: "work", label: "Work" }, { id: "home", label: "Home" }, { id: "body", label: "Body" },
  { id: "friends", label: "Friends" }, { id: "mate", label: "Mate" }, { id: "kin", label: "Kin" }, { id: "polis", label: "Polis" }, { id: "legacy", label: "Legacy" },
] as const;
export type CalTile = (typeof CAL_TILES)[number]["id"];
export const isCalTile = (v: unknown): v is CalTile => CAL_TILES.some((t) => t.id === v);
export const tileLabel = (t: CalTile | null) => CAL_TILES.find((x) => x.id === t)?.label ?? "No tile";

export const REPEATS = [
  { id: "none", label: "Once" }, { id: "daily", label: "Every day" }, { id: "weekly", label: "Every week" }, { id: "monthly", label: "Every month" }, { id: "yearly", label: "Every year" },
] as const;
export type Repeat = (typeof REPEATS)[number]["id"];
export const isRepeat = (v: unknown): v is Repeat => REPEATS.some((r) => r.id === v);

export type CalEvent = { id: string; tile: CalTile | null; title: string; note: string | null; day: string; starts: string | null; ends: string | null; repeat: Repeat; until: string | null; recordId: string | null };
export type Occurrence = { event: CalEvent; day: string };

const DAY = /^\d{4}-\d{2}-\d{2}$/;
const ms = (d: string) => Date.parse(`${d}T00:00:00Z`);
export const addDays = (d: string, n: number) => new Date(ms(d) + n * 86400_000).toISOString().slice(0, 10);
function addMonths(d: string, n: number): string | null {
  const [y, m, day] = d.split("-").map(Number) as [number, number, number];
  const t = new Date(Date.UTC(y, m - 1 + n, day));
  return t.getUTCDate() === day ? t.toISOString().slice(0, 10) : null; // the 31st skips months without one, as RFC 5545 does
}

/** Every day an event falls on from `from` through `to`, in order. */
export function occurrences(events: readonly CalEvent[], from: string, to: string): Occurrence[] {
  if (!DAY.test(from) || !DAY.test(to) || from > to) return [];
  const out: Occurrence[] = [];
  for (const e of events) {
    const last = e.until && e.until < to ? e.until : to;
    if (e.repeat === "none") {
      if (e.day >= from && e.day <= last) out.push({ event: e, day: e.day });
      continue;
    }
    for (let i = 0, d: string | null = e.day; i < 2000; i++) {
      d = e.repeat === "daily" ? addDays(e.day, i) : e.repeat === "weekly" ? addDays(e.day, 7 * i) : addMonths(e.day, e.repeat === "monthly" ? i : 12 * i);
      if (d === null) continue;
      if (d > last) break;
      if (d >= from) out.push({ event: e, day: d });
    }
  }
  return out.sort((a, b) => a.day.localeCompare(b.day) || (a.event.starts ?? "").localeCompare(b.event.starts ?? "") || a.event.title.localeCompare(b.event.title));
}

/** The Monday on or before a day, and the 42 days of a month's grid (six weeks from the Monday on or before the 1st). */
export const mondayOf = (d: string) => addDays(d, -((new Date(ms(d)).getUTCDay() + 6) % 7));
export function monthGrid(month: string): string[] {
  const first = mondayOf(`${month}-01`);
  return Array.from({ length: 42 }, (_, i) => addDays(first, i));
}
export const shiftMonth = (month: string, n: number) => addMonths(`${month}-01`, n)!.slice(0, 7);
