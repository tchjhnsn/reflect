"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import type { CatalogEntry, Sort } from "@reflect/core/openlibrary";
import { Button } from "@/components/ui/button";
import type { BookSource } from "@/server/browse";
import { BookCard } from "../book-card";
import { moreBooksAction } from "./actions";

type Saved = { books: CatalogEntry[]; mine: Record<string, string>; page: number; done: boolean; y: number };
const keyOf = () => `reflect:browse:${window.location.pathname}${window.location.search}`;
function read(): Saved | null {
  try {
    const t = window.sessionStorage.getItem(keyOf());
    return t ? (JSON.parse(t) as Saved) : null;
  } catch {
    return null;
  }
}

/**
 * More, under a list of books: the next page is added under the books already shown, and the page does not
 * reload. What was added and how far down the person was are kept for this tab, so coming back from a book
 * page shows the same list at the same place.
 */
export function MoreBooks({ source, sort, per, first }: { source: BookSource; sort?: Sort; per: number; /** How many books the page itself shows. */ first: number }) {
  const [state, setState] = useState<Omit<Saved, "y">>({ books: [], mine: {}, page: 1, done: first < per });
  const [error, setError] = useState(false);
  const [pending, run] = useTransition();
  const restored = useRef(false);

  useEffect(() => {
    const saved = read();
    if (saved?.books.length) setState({ books: saved.books, mine: saved.mine, page: saved.page, done: saved.done });
    if (saved?.y) requestAnimationFrame(() => requestAnimationFrame(() => window.scrollTo(0, saved.y)));
    restored.current = true;
  }, []);

  useEffect(() => {
    let t: ReturnType<typeof setTimeout> | undefined;
    const save = () => {
      if (!restored.current) return;
      try {
        window.sessionStorage.setItem(keyOf(), JSON.stringify({ ...state, y: window.scrollY } satisfies Saved));
      } catch {
        // storage full or blocked: the list still works, it just will not come back at the same place
      }
    };
    const onScroll = () => {
      clearTimeout(t);
      t = setTimeout(save, 150);
    };
    save();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("pagehide", save);
    return () => {
      clearTimeout(t);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("pagehide", save);
    };
  }, [state]);

  const more = () => run(async () => {
    setError(false);
    try {
      const next = await moreBooksAction(source, state.page + 1, sort);
      const seen = new Set(state.books.map((b) => b.work));
      setState({ books: [...state.books, ...next.books.filter((b) => !seen.has(b.work))], mine: { ...state.mine, ...next.mine }, page: state.page + 1, done: next.books.length < per || state.page + 1 >= 20 });
    } catch {
      setError(true);
    }
  });

  return (
    <>
      {state.books.length ? (
        <ul className="grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {state.books.map((b) => <BookCard key={b.work} b={b} have={state.mine[b.work]} />)}
        </ul>
      ) : null}
      {error ? <p className="text-muted-foreground text-center text-sm">The open library did not answer. Try again in a moment.</p> : null}
      {!state.done ? (
        <div className="flex justify-center pt-2">
          <Button variant="outline" onClick={more} disabled={pending}>{pending ? "Loading…" : "More"}</Button>
        </div>
      ) : null}
    </>
  );
}
