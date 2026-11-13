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

type Saved = { rows: Row[]; mine: [string, string][]; next: number | null; y: number };
// Leaving by a link: the place is kept as it was at the click, before the page shrinks under the next one and moves it.
let leftAt = 0;
const keyOf = () => `reflect:rows:${window.location.pathname}${window.location.search}`;

/**
 * More rows of For you as the person scrolls near the end, three at a time, until every row is shown. The rows
 * loaded and the place on the page are kept for this tab, so coming back from a book shows them where they were.
 */
export function MoreRows({ from }: { from: number }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [mine, setMine] = useState<Map<string, string>>(new Map());
  const [next, setNext] = useState<number | null>(from);
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let saved: Saved | null = null;
    try {
      saved = JSON.parse(window.sessionStorage.getItem(keyOf()) ?? "null") as Saved | null;
    } catch {
      saved = null;
    }
    if (saved && Array.isArray(saved.rows)) {
      setRows(saved.rows);
      setMine(new Map(saved.mine));
      setNext(saved.next);
      const y = saved.y;
      if (y) {
        // The rows above may still be arriving: keep trying for a moment until the page is long enough to reach y.
        let tries = 0;
        const go = () => {
          window.scrollTo(0, y);
          if (Math.abs(window.scrollY - y) > 4 && ++tries < 30) setTimeout(go, 100);
          else setReady(true);
        };
        requestAnimationFrame(go);
        return;
      }
    }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    let t: ReturnType<typeof setTimeout> | undefined;
    const save = () => {
      if (Date.now() - leftAt < 3000) return;
      try {
        window.sessionStorage.setItem(keyOf(), JSON.stringify({ rows, mine: [...mine], next, y: window.scrollY } satisfies Saved));
      } catch {
        // storage full or blocked: coming back starts from the top
      }
    };
    const onScroll = () => {
      clearTimeout(t);
      t = setTimeout(save, 200);
    };
    const onClick = (e: MouseEvent) => {
      if (!(e.target instanceof Element) || !e.target.closest("a[href]")) return;
      clearTimeout(t);
      save();
      leftAt = Date.now();
    };
    save();
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("click", onClick, true);
    return () => {
      clearTimeout(t);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("click", onClick, true);
    };
  }, [ready, rows, mine, next]);

  useEffect(() => {
    const el = end.current;
    if (!el || !ready || next === null || busy) return;
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
  }, [next, busy, ready]);

  return (
    <>
      {rows.map((r) => <BookRow key={r.key} row={r} mine={mine} />)}
      {next !== null ? <div ref={end}><RowsLoading /></div> : null}
    </>
  );
}
