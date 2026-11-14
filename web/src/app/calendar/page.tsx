import type { Metadata } from "next";
import Link from "next/link";
import { ChevronLeftIcon, ChevronRightIcon, PlusIcon, RepeatIcon } from "lucide-react";
import { CAL_TILES, isCalTile, monthGrid, occurrences, REPEATS, shiftMonth, tileLabel, type CalTile, type Occurrence } from "@reflect/core/calendar";
import { PAGE, PageHeader } from "@/components/page-header";
import { Segmented } from "@/components/segmented";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { events } from "@/server/calendar-store";
import { addEventAction, removeEventAction, restoreEventAction, saveEventAction } from "./actions";

export const metadata: Metadata = { title: "Calendar" };
export const dynamic = "force-dynamic";
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const SELECT = "border-input bg-background h-9 rounded-md border px-3 text-sm";
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const longDay = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", timeZone: "UTC" });
const monthName = (m: string) => new Date(`${m}-01T12:00:00Z`).toLocaleDateString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
const when = (o: Occurrence) => (o.event.starts ? `${o.event.starts}${o.event.ends ? `–${o.event.ends}` : ""}` : "All day");

function EventForm({ tile, day, back, e }: { tile: CalTile | null; day: string; back: string; e?: Occurrence["event"] }) {
  return (
    <form action={e ? saveEventAction.bind(null, e.id) : addEventAction} className="grid gap-3">
      <input type="hidden" name="$back" value={back} />
      <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
        <label className="grid gap-1 text-sm font-medium">What<Input name="title" required defaultValue={e?.title ?? ""} placeholder="A shift, a lecture, dinner with family" /></label>
        <label className="grid gap-1 text-sm font-medium">Tile
          <select name="tile" defaultValue={e ? e.tile ?? "" : tile ?? ""} className={SELECT}><option value="">No tile</option>{CAL_TILES.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}</select>
        </label>
      </div>
      <div className="flex flex-wrap gap-3">
        <label className="grid gap-1 text-sm font-medium">Day<Input name="day" type="date" required defaultValue={e?.day ?? day} className="w-40" /></label>
        <label className="grid gap-1 text-sm font-medium">Starts<Input name="starts" type="time" defaultValue={e?.starts ?? ""} className="w-28" /></label>
        <label className="grid gap-1 text-sm font-medium">Ends<Input name="ends" type="time" defaultValue={e?.ends ?? ""} className="w-28" /></label>
        <label className="grid gap-1 text-sm font-medium">Repeats
          <select name="repeat" defaultValue={e?.repeat ?? "none"} className={SELECT}>{REPEATS.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}</select>
        </label>
        <label className="grid gap-1 text-sm font-medium">Until<Input name="until" type="date" defaultValue={e?.until ?? ""} className="w-40" /></label>
      </div>
      <label className="grid gap-1 text-sm font-medium">Notes<Textarea name="note" rows={2} defaultValue={e?.note ?? ""} /></label>
      <div><Button type="submit" size="sm">{e ? "Save" : <><PlusIcon />Add to the calendar</>}</Button></div>
    </form>
  );
}

/**
 * The calendar, a tool (Torian, 2026-11-14 17:44Z, 17:52Z): every tile's events in one master calendar, and each
 * tile's own calendar as the same view filtered to its tag (`?tile=work`).
 */
export default async function CalendarPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const today = new Date().toISOString().slice(0, 10);
  const tile = isCalTile(one(sp.tile)) ? (one(sp.tile) as CalTile) : null;
  const picked = /^\d{4}-\d{2}-\d{2}$/.test(one(sp.day)) ? one(sp.day) : null;
  const month = /^\d{4}-\d{2}$/.test(one(sp.month)) ? one(sp.month) : (picked ?? today).slice(0, 7);
  const grid = monthGrid(month);
  const list = occurrences(await events(user.id, grid[0]!, grid[41]!, tile), grid[0]!, grid[41]!);
  const byDay = new Map<string, Occurrence[]>();
  for (const o of list) byDay.set(o.day, [...(byDay.get(o.day) ?? []), o]);
  const q = (over: Record<string, string | null>) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries({ tile, month, day: picked, ...over })) if (v) p.set(k, v);
    const s = p.toString();
    return `/calendar${s ? `?${s}` : ""}`;
  };
  const back = q({});
  const removed = one(sp.removed);
  const inMonth = list.filter((o) => o.day.startsWith(month));
  const day = picked ?? (month === today.slice(0, 7) ? today : `${month}-01`);
  const dayList = byDay.get(day) ?? [];
  return (
    <main className={PAGE}>
      <PageHeader back="/" backLabel="Home" title={tile ? `${tileLabel(tile)} calendar` : "Calendar"} />
      <div className="mt-2 grid grid-cols-[minmax(0,1fr)] gap-6">
        <div className="-mx-1 overflow-x-auto px-1">
          <Segmented label="Whose calendar" items={[{ label: "All", href: q({ tile: null }), on: !tile }, ...CAL_TILES.map((t) => ({ label: t.label, href: q({ tile: t.id }), on: tile === t.id }))]} />
        </div>
        {removed ? (
          <form action={restoreEventAction.bind(null, removed)} className="bg-muted flex items-center justify-between gap-3 rounded-lg px-4 py-2 text-sm">
            <input type="hidden" name="$back" value={back} />Removed from the calendar.<Button type="submit" variant="outline" size="sm">Undo</Button>
          </form>
        ) : null}
        <div className="flex items-center gap-2">
          <Button asChild variant="ghost" size="icon" aria-label="Last month"><Link href={q({ month: shiftMonth(month, -1), day: null })}><ChevronLeftIcon /></Link></Button>
          <h2 className="min-w-44 text-center text-lg font-semibold">{monthName(month)}</h2>
          <Button asChild variant="ghost" size="icon" aria-label="Next month"><Link href={q({ month: shiftMonth(month, 1), day: null })}><ChevronRightIcon /></Link></Button>
          {month !== today.slice(0, 7) ? <Button asChild variant="outline" size="sm"><Link href={q({ month: null, day: null })}>Today</Link></Button> : null}
        </div>

        <div className="hidden overflow-hidden rounded-xl border md:block">
          <div className="text-muted-foreground grid grid-cols-7 border-b text-xs">{WEEKDAYS.map((w) => <div key={w} className="px-2 py-1.5">{w}</div>)}</div>
          <div className="grid grid-cols-7">
            {grid.map((d) => {
              const os = byDay.get(d) ?? [];
              return (
                <Link key={d} href={q({ day: d, month })} scroll={false} className={cn("hover:bg-accent min-h-24 border-r border-b p-1.5 text-xs [&:nth-child(7n)]:border-r-0", !d.startsWith(month) && "text-muted-foreground/60", d === day && "bg-accent")}>
                  <span className={cn("inline-flex size-6 items-center justify-center rounded-full", d === today && "bg-foreground text-background")}>{Number(d.slice(8))}</span>
                  <ul className="mt-1 grid gap-0.5">
                    {os.slice(0, 3).map((o) => <li key={o.event.id} className="truncate"><span className="text-muted-foreground">{o.event.starts ?? ""} </span>{o.event.title}</li>)}
                    {os.length > 3 ? <li className="text-muted-foreground">{os.length - 3} more</li> : null}
                  </ul>
                </Link>
              );
            })}
          </div>
        </div>

        <section className="grid gap-3 md:hidden">
          {inMonth.length ? (
            [...new Set(inMonth.map((o) => o.day))].map((d) => (
              <div key={d} className="grid gap-1">
                <Link href={q({ day: d })} className={cn("text-sm font-semibold", d === today && "underline underline-offset-4")}>{longDay(d)}</Link>
                <ul className="divide-y rounded-lg border text-sm">{byDay.get(d)!.map((o) => <li key={o.event.id} className="flex gap-3 px-3 py-2"><span className="text-muted-foreground w-20 shrink-0 text-xs">{when(o)}</span><span className="min-w-0">{o.event.title}{!tile && o.event.tile ? <span className="text-muted-foreground"> · {tileLabel(o.event.tile)}</span> : null}</span></li>)}</ul>
              </div>
            ))
          ) : <p className="text-muted-foreground text-sm">Nothing on the calendar this month.</p>}
        </section>

        <section className="grid gap-3">
          <h2 className="text-lg font-semibold">{longDay(day)}</h2>
          {dayList.length ? (
            <ul className="grid gap-2">
              {dayList.map((o) => (
                <li key={o.event.id}>
                  <details className="rounded-lg border">
                    <summary className="flex cursor-pointer items-baseline gap-3 px-4 py-2 text-sm">
                      <span className="text-muted-foreground w-24 shrink-0 text-xs">{when(o)}</span>
                      <span className="min-w-0 flex-1 font-medium">{o.event.title}</span>
                      {o.event.repeat !== "none" ? <RepeatIcon className="text-muted-foreground size-3.5" aria-label={REPEATS.find((r) => r.id === o.event.repeat)!.label} /> : null}
                      <span className="text-muted-foreground text-xs">{tileLabel(o.event.tile)}</span>
                    </summary>
                    <div className="grid gap-3 border-t p-4">
                      {o.event.note ? <p className="text-sm whitespace-pre-line">{o.event.note}</p> : null}
                      <EventForm tile={tile} day={day} back={back} e={o.event} />
                      <form action={removeEventAction.bind(null, o.event.id)}><input type="hidden" name="$back" value={back} /><Button type="submit" variant="ghost" size="sm">Remove</Button></form>
                    </div>
                  </details>
                </li>
              ))}
            </ul>
          ) : <p className="text-muted-foreground text-sm">Nothing on this day{tile ? ` in ${tileLabel(tile)}` : ""}.</p>}
          <details className="rounded-xl border p-4" open={!list.length}>
            <summary className="cursor-pointer font-semibold">Add an event</summary>
            <div className="mt-4"><EventForm tile={tile} day={day} back={back} /></div>
          </details>
        </section>
      </div>
    </main>
  );
}
