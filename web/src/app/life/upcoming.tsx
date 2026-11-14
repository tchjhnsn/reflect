import Link from "next/link";
import { addDays, occurrences, tileLabel, type CalTile } from "@reflect/core/calendar";
import { events } from "@/server/calendar-store";

const shortDay = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });

/** A tile's calendar for the next two weeks, from the events tagged with it. */
export async function Upcoming({ userId, tile, today }: { userId: string; tile: CalTile; today: string }) {
  const to = addDays(today, 13);
  const list = occurrences(await events(userId, today, to, tile), today, to).slice(0, 8);
  return (
    <section className="grid gap-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Coming up</h2>
          <p className="text-muted-foreground text-sm">The next two weeks on your {tileLabel(tile)} calendar.</p>
        </div>
        <Link href={`/calendar?tile=${tile}`} className="text-muted-foreground hover:text-foreground text-sm underline-offset-2 hover:underline">Open the calendar</Link>
      </div>
      {list.length ? (
        <ul className="divide-y rounded-xl border text-sm">
          {list.map((o) => (
            <li key={`${o.event.id}-${o.day}`}>
              <Link href={`/calendar?tile=${tile}&day=${o.day}`} className="hover:bg-accent flex items-baseline gap-3 px-4 py-2">
                <span className="text-muted-foreground w-24 shrink-0 text-xs">{shortDay(o.day)}{o.event.starts ? ` ${o.event.starts}` : ""}</span>
                <span className="min-w-0 truncate font-medium">{o.event.title}</span>
              </Link>
            </li>
          ))}
        </ul>
      ) : <p className="text-muted-foreground text-sm">Nothing on it. <Link href={`/calendar?tile=${tile}&day=${today}`} className="underline underline-offset-2">Add an event</Link> tagged {tileLabel(tile)}.</p>}
    </section>
  );
}
