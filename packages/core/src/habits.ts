// Reading habits from the days a person read: the run of days up to today, and a calendar of weeks. Pure.

const DAY_MS = 86400000;
const iso = (t: number) => new Date(t).toISOString().slice(0, 10);
const at = (d: string) => Date.parse(`${d}T00:00:00Z`);

/** Days in a row read, ending today or yesterday (a day not yet read does not break the run). */
export function streak(days: readonly string[], today: string): number {
  const set = new Set(days);
  let t = at(today);
  if (!set.has(iso(t))) t -= DAY_MS;
  let n = 0;
  while (set.has(iso(t))) {
    n++;
    t -= DAY_MS;
  }
  return n;
}

/**
 * The last `weeks` weeks to today as columns of seven days, Monday first, each day with whether it was read;
 * days after today are null.
 */
export function calendar(days: readonly string[], today: string, weeks = 26): ({ day: string; read: boolean } | null)[][] {
  const set = new Set(days);
  const t = at(today);
  const weekday = (new Date(t).getUTCDay() + 6) % 7;
  const start = t - weekday * DAY_MS - (weeks - 1) * 7 * DAY_MS;
  return Array.from({ length: weeks }, (_, w) =>
    Array.from({ length: 7 }, (_, d) => {
      const x = start + (w * 7 + d) * DAY_MS;
      return x > t ? null : { day: iso(x), read: set.has(iso(x)) };
    }),
  );
}

/** Books finished in a year, by month (1 to 12), in the order finished. */
export function byMonth<T extends { finishedOn: string | null }>(books: readonly T[], year: number): Map<number, T[]> {
  const out = new Map<number, T[]>();
  for (const b of [...books].filter((x) => x.finishedOn?.startsWith(`${year}-`)).sort((a, c) => a.finishedOn!.localeCompare(c.finishedOn!))) {
    const m = Number(b.finishedOn!.slice(5, 7));
    out.set(m, [...(out.get(m) ?? []), b]);
  }
  return out;
}
