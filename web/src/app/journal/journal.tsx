"use client";

import { byDay, onThisDay, type Entry } from "@/core/journal";
import type { IsoDate } from "@/core/person";
import { Composer } from "./composer";
import { EntryCard } from "./entry-card";
import { Nav } from "../nav";
import { useToday } from "@/lib/today";

function dayLabel(day: IsoDate, today: IsoDate): string {
  if (day === today) return "Today";
  const d = new Date(`${day}T12:00:00`);
  const y = new Date(`${today}T12:00:00`);
  y.setDate(y.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return "Yesterday";
  return d.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: day.slice(0, 4) === today.slice(0, 4) ? undefined : "numeric" });
}

export function Journal({ name, entries, packs, used }: { name: string | null; entries: Entry[]; packs: string[]; used: string[] }) {
  const today = useToday();
  const past = today ? onThisDay(entries, today) : [];

  return (
    <main className="shell wide">
      <Nav current="journal" />
      <h1 className="sr-only">{name ? `${name}'s journal` : "Your journal"}</h1>
      {today ? <Composer today={today} packs={packs} used={used} /> : <div className="composer placeholder" />}
      {past.length > 0 && today ? (
        <section className="on-this-day" aria-label="On this day">
          <h2 className="day">On this day</h2>
          {past.map((e) => (
            <EntryCard key={e.id} entry={e} label={new Date(`${e.localDate}T12:00:00`).getFullYear().toString()} />
          ))}
        </section>
      ) : null}
      {entries.length === 0 ? (
        <p className="note empty" data-testid="empty">
          {name ? `Nothing here yet, ${name}.` : "Nothing here yet."} Whatever you write stays private to you.
        </p>
      ) : (
        byDay(entries).map(({ day, entries: list }) => (
          <section key={day} aria-label={today ? dayLabel(day, today) : day}>
            <h2 className="day">{today ? dayLabel(day, today) : day}</h2>
            {list.map((e) => (
              <EntryCard key={e.id} entry={e} />
            ))}
          </section>
        ))
      )}
    </main>
  );
}
