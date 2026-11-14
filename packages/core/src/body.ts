import { fitnessWeek, type ActFacts, type RecordFacts } from "./life";

const DAY = 86400_000;
const since = (today: string, days: number) => new Date(Date.parse(`${today}T00:00:00Z`) - (days - 1) * DAY).toISOString().slice(0, 10);
const ACTIVE = new Set(["moderate", "vigorous", "strength"]);

export type GoalView = { id: string; goal: string; by: string | null; state: string | null; minutes: number; last: string | null };

/**
 * Body's workspace from the person's own records: the week's activity against the guidelines, sleep against seven
 * hours (the hours they usually sleep, and the nights they logged this week), their last checkup and how long ago,
 * and each fitness goal with the minutes put toward it in the last four weeks.
 */
export function bodySpace(records: readonly (RecordFacts & { id: string })[], acts: readonly ActFacts[], today: string) {
  const h = records.find((r) => r.kind === "health");
  const week = fitnessWeek(acts, today);
  const from = since(today, 7);
  const nights = acts.filter((a) => a.kind === "slept" && a.day >= from && a.day <= today && a.minutes);
  const logged = nights.length ? Math.round((nights.reduce((s, a) => s + a.minutes!, 0) / nights.length / 60) * 10) / 10 : null;
  const checkup = typeof h?.body.checkup === "string" ? h.body.checkup : null;
  const four = since(today, 28);
  const goals: GoalView[] = records.filter((r) => r.kind === "fitness" && r.state !== "reached" && r.state !== "lapsed").map((r) => {
    const mine = acts.filter((a) => a.record === r.id && ACTIVE.has(a.kind));
    return {
      id: r.id, goal: String(r.body.goal ?? ""), by: typeof r.body.by === "string" ? r.body.by : null, state: r.state,
      minutes: mine.filter((a) => a.day >= four && a.day <= today).reduce((s, a) => s + (a.minutes ?? 0), 0),
      last: mine.map((a) => a.day).sort().at(-1) ?? null,
    };
  });
  return {
    week,
    sleep: { usual: typeof h?.body.sleep === "number" ? h.body.sleep : null, logged, nights: nights.length },
    health: { coverage: typeof h?.body.coverage === "string" && h.body.coverage ? h.body.coverage : null, doctor: typeof h?.body.doctor === "string" && h.body.doctor ? h.body.doctor : null },
    checkup: checkup ? { day: checkup, days: Math.round((Date.parse(`${today}T00:00:00Z`) - Date.parse(`${checkup}T00:00:00Z`)) / DAY) } : null,
    healthId: h?.id ?? null,
    goals,
  };
}
