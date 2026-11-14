import Link from "next/link";
import { bodySpace } from "@reflect/core/body";
import type { LifeAct, LifeRecord } from "@/server/life-store";
import { Head, SPACE } from "./space";
import { Upcoming } from "./upcoming";

const ago = (days: number) => (days < 1 ? "today" : days < 45 ? `${days} ${days === 1 ? "day" : "days"} ago` : days < 365 ? `${Math.round(days / 30)} months ago` : `${Math.floor(days / 365)} ${Math.floor(days / 365) === 1 ? "year" : "years"} ago`);
const BOX = "rounded-xl border p-4";

/**
 * Body's workspace (Torian, 2026-11-14 17:44Z, 18:11Z): the person's week against the activity guidelines and
 * seven hours of sleep, their coverage, doctor and last checkup, the fitness goals under way with the time put
 * toward each, and Body's calendar.
 */
export async function BodySpace({ userId, records, acts, today }: { userId: string; records: LifeRecord[]; acts: LifeAct[]; today: string }) {
  const s = bodySpace(records, acts, today);
  const sleep = s.sleep.logged ?? s.sleep.usual;
  return (
    <div className={SPACE}>
      <section className="grid gap-3">
        <Head title="This week" note="The last seven days against the guidelines: 150 minutes of activity, strength on two days, seven hours of sleep." />
        <dl className="grid gap-3 sm:grid-cols-3">
          <div className={BOX}><dt className="text-muted-foreground text-xs">Activity</dt><dd className="mt-1 text-2xl font-semibold">{s.week.minutes} <span className="text-muted-foreground text-base font-normal">of 150 minutes</span></dd><dd className="text-muted-foreground mt-1 text-xs">Vigorous minutes count twice.</dd></div>
          <div className={BOX}><dt className="text-muted-foreground text-xs">Strength</dt><dd className="mt-1 text-2xl font-semibold">{s.week.strengthDays} <span className="text-muted-foreground text-base font-normal">of 2 days</span></dd></div>
          <div className={BOX}><dt className="text-muted-foreground text-xs">Sleep</dt><dd className="mt-1 text-2xl font-semibold">{sleep !== null ? <>{sleep} <span className="text-muted-foreground text-base font-normal">of 7 hours</span></> : "Not known"}</dd><dd className="text-muted-foreground mt-1 text-xs">{s.sleep.logged !== null ? `Average of ${s.sleep.nights} ${s.sleep.nights === 1 ? "night" : "nights"} logged this week.` : s.sleep.usual !== null ? "What you usually sleep, from Health." : <>Say what you usually sleep in <Link href="/life/body?k=health" className="underline underline-offset-2">Health</Link>.</>}</dd></div>
        </dl>
      </section>

      <section className="grid gap-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <Head title="Goals under way" note="Your own goals, with the minutes you put toward each in the last four weeks." />
          <Link href="/life/body?k=fitness" className="text-muted-foreground hover:text-foreground text-sm underline-offset-2 hover:underline">Set a goal</Link>
        </div>
        {s.goals.length ? (
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {s.goals.map((g) => (
              <li key={g.id}>
                <Link href={`/life/body/${g.id}`} className="bg-card hover:bg-accent flex h-full flex-col gap-2 rounded-xl border p-4 shadow-xs transition">
                  <span className="font-medium leading-snug">{g.goal || "A goal"}</span>
                  <span className="text-sm">{g.minutes} minutes in four weeks</span>
                  <span className="text-muted-foreground mt-auto text-xs">{[g.by ? `By ${g.by}` : null, g.last ? `last on ${g.last}` : "Nothing logged toward it yet"].filter(Boolean).join(", ")}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : <p className="text-muted-foreground text-sm">No goal under way. <Link href="/life/body?k=fitness" className="underline underline-offset-2">Set one</Link> beyond the guidelines: run a 5K, squat your bodyweight.</p>}
      </section>

      <section className="grid gap-3">
        <Head title="Health" note="Coverage, a regular doctor, and your last checkup." />
        <Link href="/life/body?k=health" className="hover:bg-accent grid gap-3 rounded-xl border p-4 text-sm transition sm:grid-cols-3">
          <span><span className="text-muted-foreground block text-xs">Coverage</span>{s.health.coverage ?? "Not stated"}</span>
          <span><span className="text-muted-foreground block text-xs">Regular doctor</span>{s.health.doctor ?? "Not stated"}</span>
          <span><span className="text-muted-foreground block text-xs">Last checkup</span>{s.checkup ? `${s.checkup.day}, ${ago(s.checkup.days)}` : "Not stated"}</span>
        </Link>
      </section>

      <Upcoming userId={userId} tile="body" today={today} />
    </div>
  );
}
