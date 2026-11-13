"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { ArrowDownIcon, ArrowUpIcon, GripVerticalIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { rankWantAction, setFinishedAction, setGoalAction } from "../actions";

/** A reading goal for a year: books finished against the goal, and a field to set, change or clear it. */
export function Goal({ year, goal, done }: { year: number; goal: number | null; done: number }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(String(goal ?? 12));
  const [pending, run] = useTransition();
  const save = (books: number | null) => run(async () => {
    await setGoalAction(year, books);
    setEditing(false);
  });
  if (editing) {
    return (
      <form className="flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); const n = Number(value); if (Number.isInteger(n) && n >= 1) save(n); }}>
        <Input type="number" min={1} max={1000} value={value} onChange={(e) => setValue(e.target.value)} className="h-9 w-24" aria-label={`Books to read in ${year}`} />
        <span className="text-sm">books in {year}</span>
        <Button size="sm" type="submit" disabled={pending}>Save</Button>
        {goal ? <Button size="sm" type="button" variant="ghost" disabled={pending} onClick={() => save(null)}>No goal</Button> : null}
        <Button size="sm" type="button" variant="ghost" onClick={() => setEditing(false)}>Cancel</Button>
      </form>
    );
  }
  if (!goal) return <div className="flex flex-wrap items-center gap-3"><p className="text-sm">{done} {done === 1 ? "book" : "books"} finished in {year}.</p><Button size="sm" variant="outline" onClick={() => setEditing(true)}>Set a goal for {year}</Button></div>;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm"><span className="text-2xl font-semibold tabular-nums">{done}</span> of {goal} books in {year}{done >= goal ? ", goal reached" : ""}</p>
        <Button size="sm" variant="ghost" onClick={() => setEditing(true)}>Change</Button>
      </div>
      <div className="bg-border h-2 overflow-hidden rounded-full"><div className="bg-foreground/70 h-full" style={{ width: `${Math.min(100, (done / goal) * 100)}%` }} /></div>
    </div>
  );
}

/** Want to read in the person's order: drag a book, or move it with its arrows, and the order is kept. */
export function WantList({ books }: { books: { id: string; title: string; author: string; work: string | null; cover: string | null }[] }) {
  const [order, setOrder] = useState(books);
  const [drag, setDrag] = useState<number | null>(null);
  const [pending, run] = useTransition();
  const commit = (next: typeof books) => {
    setOrder(next);
    run(() => rankWantAction(next.map((b) => b.id)));
  };
  const move = (from: number, to: number) => {
    if (to < 0 || to >= order.length || from === to) return;
    const next = [...order];
    const [b] = next.splice(from, 1);
    next.splice(to, 0, b!);
    commit(next);
  };
  if (!order.length) return <p className="text-muted-foreground text-sm">Nothing on Want to read. Add books from <Link href="/learn/browse" className="underline underline-offset-2">Browse</Link>.</p>;
  return (
    <ol className="divide-y rounded-md border" aria-busy={pending}>
      {order.map((b, i) => (
        <li
          key={b.id}
          draggable
          onDragStart={() => setDrag(i)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={() => { if (drag !== null) move(drag, i); setDrag(null); }}
          onDragEnd={() => setDrag(null)}
          className={cn("flex items-center gap-3 px-2 py-2", drag === i && "opacity-50")}
        >
          <GripVerticalIcon className="text-muted-foreground size-4 shrink-0 cursor-grab" aria-hidden />
          <span className="text-muted-foreground w-6 text-right text-xs tabular-nums">{i + 1}</span>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {b.cover ? <img src={b.cover} alt="" className="h-10 w-7 rounded-sm object-cover" /> : <span className="bg-muted h-10 w-7 rounded-sm" />}
          <span className="min-w-0 flex-1">
            {b.work ? <Link href={`/learn/works/${b.work}`} className="block truncate text-sm font-medium hover:underline">{b.title}</Link> : <span className="block truncate text-sm font-medium">{b.title}</span>}
            <span className="text-muted-foreground block truncate text-xs">{b.author}</span>
          </span>
          <Button size="icon" variant="ghost" className="size-7" aria-label={`Move ${b.title} up`} disabled={i === 0} onClick={() => move(i, i - 1)}><ArrowUpIcon /></Button>
          <Button size="icon" variant="ghost" className="size-7" aria-label={`Move ${b.title} down`} disabled={i === order.length - 1} onClick={() => move(i, i + 1)}><ArrowDownIcon /></Button>
        </li>
      ))}
    </ol>
  );
}

/** The day a book was finished, changed in place. */
export function FinishedDay({ bookId, day, max }: { bookId: string; day: string; max: string }) {
  const [value, setValue] = useState(day);
  const [pending, run] = useTransition();
  return (
    <input
      type="date"
      value={value}
      max={max}
      min="1900-01-01"
      disabled={pending}
      aria-label="Finished on"
      className="text-muted-foreground bg-transparent text-xs"
      onChange={(e) => {
        const v = e.target.value;
        setValue(v);
        if (v) run(() => setFinishedAction(bookId, v));
      }}
    />
  );
}
