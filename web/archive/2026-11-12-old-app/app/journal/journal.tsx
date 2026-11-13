"use client";

import { byDay, onThisDay, type Entry } from "@reflect/core/journal";
import type { IsoDate } from "@reflect/core/person";
import { useToday } from "@/lib/today";
import { Composer } from "./composer";
import { EntryCard } from "./entry-card";
import { Nav, initialOf } from "../nav";

const at = (day: string) => new Date(`${day}T12:00:00`);

function dayLabel(day: IsoDate, today: IsoDate): string {
  if (day === today) return "Today";
  const y = at(today);
  y.setDate(y.getDate() - 1);
  if (at(day).toDateString() === y.toDateString()) return "Yesterday";
  return at(day).toLocaleDateString(undefined, { weekday: "short", month: "short", year: day.slice(0, 4) === today.slice(0, 4) ? undefined : "numeric" });
}

function greeting(): string {
  const h = new Date().getHours();
  return h < 5 ? "Good night" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

export function Journal({ name, entries, packs, used }: { name: string | null; entries: Entry[]; packs: string[]; used: string[] }) {
  const today = useToday();
  const past = today ? onThisDay(entries, today) : [];
  const eyebrow = today ? at(today).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" }) : " ";

  return (
    <main className="page">
      <Nav current="journal" eyebrow={eyebrow} initial={initialOf(name)} />
      <h1 className="title">
        {today ? greeting() : "Hello"}
        {name ? (
          <>
            , <em>{name}</em>.
          </>
        ) : (
          "."
        )}
      </h1>
      {today ? <Composer today={today} packs={packs} used={used} /> : <div className="composer placeholder" />}
      {past.length > 0 && today ? (
        <section className="on-this-day" aria-label="On this day">
          {past.map((e) => (
            <div key={e.id}>
              <h2 className="lbl" style={{ color: "var(--accent)" }}>
                On this day, {e.localDate.slice(0, 4)}
              </h2>
              <EntryCard entry={e} label={e.localDate.slice(0, 4)} />
            </div>
          ))}
        </section>
      ) : null}
      {entries.length === 0 ? (
        <p className="empty" data-testid="empty">
          {name ? `Nothing here yet, ${name}.` : "Nothing here yet."} Whatever you write stays private to you.
        </p>
      ) : (
        byDay(entries).map(({ day, entries: list }) => {
          const label = today ? dayLabel(day, today) : day;
          return (
            <section key={day} className="day" aria-label={label}>
              <h2 className="gutter">
                <b aria-hidden="true">{Number(day.slice(8, 10))}</b>
                <span>{label}</span>
              </h2>
              <div className="posts">
                {list.map((e) => (
                  <EntryCard key={e.id} entry={e} />
                ))}
              </div>
            </section>
          );
        })
      )}
    </main>
  );
}
