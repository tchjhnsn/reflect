import type { Metadata } from "next";
import Link from "next/link";
import { byMonth, calendar, streak } from "@reflect/core/habits";
import { coverUrl } from "@reflect/core/openlibrary";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { goalFor, learnView, readingDays } from "@/server/learn-store";
import { LearnHeader } from "../header";
import { FinishedDay, Goal, WantList } from "./parts";

export const metadata: Metadata = { title: "Reading" };
export const dynamic = "force-dynamic";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * The person's reading: what they are reading now, a goal for the year if they set one, the days they read in
 * Reflect, Want to read in their own order, and the books they finished, year by year, each with the day.
 */
export default async function ReadingPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const today = new Date().toISOString().slice(0, 10);
  const thisYear = Number(today.slice(0, 4));
  const view = await learnView(user.id);
  const finished = view.books.filter((b) => b.status === "read" && b.finishedOn);
  const years = [...new Set([thisYear, ...finished.map((b) => Number(b.finishedOn!.slice(0, 4)))])].sort((a, b) => b - a);
  const asked = Number(one((await searchParams).y));
  const year = years.includes(asked) ? asked : thisYear;
  const from = `${thisYear - 1}-${today.slice(5)}`;
  const [goal, days] = await Promise.all([goalFor(user.id, year), readingDays(user.id, from, today)]);
  const months = byMonth(finished, year);
  const doneInYear = [...months.values()].reduce((n, l) => n + l.length, 0);
  const reading = view.books.filter((b) => b.status === "reading");
  const want = view.books.filter((b) => b.status === "want").sort((a, b) => (a.wantRank ?? 1e9) - (b.wantRank ?? 1e9));
  const run = streak(days, today);
  const weeks = calendar(days, today, 26);
  const thisYearDays = days.filter((d) => d.startsWith(`${thisYear}-`)).length;
  const cover = (b: (typeof view.books)[number]) => (b.catalog ? coverUrl(b.catalog, "S") : null);

  return (
    <>
      <LearnHeader on="reading" />
      <main className="mx-auto w-full max-w-3xl space-y-12 px-4 pt-4 pb-24 sm:px-6">
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Reading now</h2>
          {reading.length ? (
            <ul className="grid gap-3 sm:grid-cols-2">
              {reading.map((b) => (
                <li key={b.id} className="flex items-center gap-3 rounded-md border p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  {cover(b) ? <img src={cover(b)!} alt="" className="h-14 w-10 rounded-sm object-cover" /> : <span className="bg-muted h-14 w-10 rounded-sm" />}
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">{b.title}</span>
                    <span className="text-muted-foreground block truncate text-xs">{b.authors[0] ?? ""}{b.place ? ` · chapter ${b.place.chapter} of ${b.place.chapters}` : ""}</span>
                    {b.catalog?.work ? <Link href={`/learn/works/${b.catalog.work}${b.place ? "/read" : ""}`} className="text-xs underline underline-offset-2">{b.place ? "Continue reading" : "Open the book"}</Link> : null}
                  </span>
                </li>
              ))}
            </ul>
          ) : <p className="text-muted-foreground text-sm">No book marked Reading.</p>}
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Goal</h2>
          <Goal year={year} goal={goal} done={doneInYear} />
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Days you read</h2>
          <p className="text-muted-foreground text-sm">{thisYearDays} {thisYearDays === 1 ? "day" : "days"} in {thisYear}{run > 1 ? `, the last ${run} in a row` : ""}. A day counts when you read a book on your shelves in Reflect.</p>
          <div className="overflow-x-auto">
            <div className="flex gap-1" role="img" aria-label={`Days read over the last 26 weeks: ${days.filter((d) => d >= weeks[0]![0]!.day).length}`}>
              {weeks.map((w, i) => (
                <div key={i} className="flex flex-col gap-1">
                  {w.map((d, j) => <span key={j} title={d ? `${d.day}${d.read ? ", read" : ""}` : undefined} className={cn("size-3 rounded-sm", !d ? "bg-transparent" : d.read ? "bg-foreground/70" : "bg-muted")} />)}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Want to read</h2>
          <WantList books={want.map((b) => ({ id: b.id, title: b.title, author: b.authors[0] ?? "", work: b.catalog?.work ?? null, cover: cover(b) }))} />
        </section>

        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold">Your year in books, {year}</h2>
            {years.length > 1 ? (
              <nav className="flex flex-wrap gap-1 text-sm" aria-label="Year">
                {years.map((y) => <Link key={y} href={`/learn/reading?y=${y}`} aria-current={y === year ? "page" : undefined} className={cn("rounded-full border px-3 py-1", y === year && "bg-muted")}>{y}</Link>)}
              </nav>
            ) : null}
          </div>
          {doneInYear ? (
            <>
              <p className="text-sm">{doneInYear} {doneInYear === 1 ? "book" : "books"} finished in {year}.</p>
              <div className="space-y-6">
                {[...months.entries()].map(([m, list]) => (
                  <div key={m} className="space-y-2">
                    <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">{MONTHS[m - 1]}</h3>
                    <ul className="space-y-2">
                      {list.map((b) => (
                        <li key={b.id} className="flex items-center gap-3">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          {cover(b) ? <img src={cover(b)!} alt="" className="h-12 w-8 rounded-sm object-cover" /> : <span className="bg-muted h-12 w-8 rounded-sm" />}
                          <span className="min-w-0 flex-1">
                            {b.catalog?.work ? <Link href={`/learn/works/${b.catalog.work}`} className="block truncate text-sm font-medium hover:underline">{b.title}</Link> : <span className="block truncate text-sm font-medium">{b.title}</span>}
                            <span className="text-muted-foreground block truncate text-xs">{b.authors[0] ?? ""}</span>
                          </span>
                          <FinishedDay bookId={b.id} day={b.finishedOn!} max={today} />
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </>
          ) : <p className="text-muted-foreground text-sm">No books finished in {year} yet. When you mark a book Read, the day is kept here, and you can change it.</p>}
        </section>
      </main>
    </>
  );
}
