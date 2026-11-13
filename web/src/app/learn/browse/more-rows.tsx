"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Row } from "@/server/rows";
import { BookGrid } from "../book-card";
import { moreRowsAction } from "./actions";

/** A row of books as For you shows it: a title, See all, and the books, one row to swipe on a phone. */
export function BookRow({ row, mine }: { row: Row; mine: Map<string, string> }) {
  return (
    <section className="space-y-4">
      <div className="flex items-baseline justify-between gap-3 pt-2">
        <h2 className="text-lg font-semibold">{row.title}</h2>
        <Link href={row.href} className="text-muted-foreground shrink-0 text-sm underline-offset-4 hover:underline">See all</Link>
      </div>
      <BookGrid books={row.books} mine={mine} row />
    </section>
  );
}

/** Placeholder rows while the next ones load. */
export function RowsLoading({ n = 2 }: { n?: number }) {
  return (
    <div className="space-y-8" aria-label="Loading more books">
      {Array.from({ length: n }, (_, i) => (
        <div key={i} className="space-y-4 pt-2">
          <div className="bg-muted h-5 w-56 animate-pulse rounded" />
          <div className="grid grid-cols-3 gap-4 md:grid-cols-4 lg:grid-cols-6">
            {Array.from({ length: 6 }, (_, k) => <div key={k} className={`bg-muted aspect-[2/3] animate-pulse rounded-md ${k >= 3 ? "hidden md:block" : ""} ${k >= 4 ? "md:hidden lg:block" : ""}`} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

/** More rows of For you as the person scrolls near the end, three at a time, until every row is shown. */
export function MoreRows({ from }: { from: number }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [mine, setMine] = useState<Map<string, string>>(new Map());
  const [next, setNext] = useState<number | null>(from);
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = end.current;
    if (!el || next === null || busy) return;
    const io = new IntersectionObserver((seen) => {
      if (!seen.some((s) => s.isIntersecting)) return;
      io.disconnect();
      setBusy(true);
      moreRowsAction(next)
        .then((r) => {
          setRows((x) => [...x, ...r.rows.filter((row) => !x.some((y) => y.key === row.key))]);
          setMine((m) => new Map([...m, ...Object.entries(r.mine)]));
          setNext(r.next);
        })
        .catch(() => setNext(null))
        .finally(() => setBusy(false));
    }, { rootMargin: "800px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [next, busy]);

  return (
    <>
      {rows.map((r) => <BookRow key={r.key} row={r} mine={mine} />)}
      {next !== null ? <div ref={end}><RowsLoading /></div> : null}
    </>
  );
}
