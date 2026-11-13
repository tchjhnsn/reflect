"use client";

import Link from "next/link";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { CheckIcon, ListIcon, MinusIcon, PlusIcon, Settings2Icon } from "lucide-react";
import { readingTime, minutesFor, timeLeft } from "@reflect/core/reading";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

/** How the person likes to read: kept in this browser. */
export type ReaderLook = { font: "serif" | "sans"; size: number; spacing: "compact" | "normal" | "wide"; theme: "light" | "sepia" | "dark"; turn: "scroll" | "pages" };
const LOOK: ReaderLook = { font: "serif", size: 19, spacing: "normal", theme: "light", turn: "scroll" };
const KEY = "reflect:reader";
const LEADING = { compact: 1.5, normal: 1.75, wide: 2.05 } as const;
const SIZES = [15, 17, 19, 21, 23, 26, 29];

type Reader = { look: ReaderLook; setLook: (l: Partial<ReaderLook>) => void; at: number; setAt: (a: number) => void };
const ReaderContext = createContext<Reader>({ look: LOOK, setLook: () => undefined, at: 0, setAt: () => undefined });
export const useReader = () => useContext(ReaderContext);

/**
 * The page a book is read on: the person's font, size, spacing and light, sepia or dark, applied to everything
 * inside it, and how far through the chapter they are, shared with the contents and the time left.
 */
export function ReaderFrame({ children }: { children: React.ReactNode }) {
  const [look, set] = useState<ReaderLook>(LOOK);
  const [at, setAt] = useState(0);
  useEffect(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem(KEY) ?? "null") as Partial<ReaderLook> | null;
      if (saved) set({ ...LOOK, ...saved });
    } catch {
      // storage blocked: the default look
    }
  }, []);
  const value = useMemo<Reader>(() => ({
    look,
    at,
    setAt,
    setLook: (l) => set((old) => {
      const next = { ...old, ...l };
      try {
        window.localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        // storage blocked: the look holds until the page is left
      }
      return next;
    }),
  }), [look, at]);
  const style = { "--reader-size": `${look.size}px`, "--reader-leading": String(LEADING[look.spacing]) } as React.CSSProperties;
  return (
    <ReaderContext.Provider value={value}>
      <div style={style} data-font={look.font} className={cn("reader bg-background text-foreground min-h-svh", look.theme === "dark" && "dark", look.theme === "sepia" && "reader-sepia")}>
        {children}
      </div>
    </ReaderContext.Provider>
  );
}

function Choice<T extends string>({ value, options, onPick }: { value: T; options: { id: T; label: string }[]; onPick: (v: T) => void }) {
  return (
    <div className="grid gap-1 px-2 pb-2" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map((o) => (
        <Button key={o.id} size="sm" variant={o.id === value ? "default" : "outline"} onClick={() => onPick(o.id)}>{o.label}</Button>
      ))}
    </div>
  );
}

/** The reading settings: font, size, spacing, light, sepia or dark, and scrolling or turning pages. */
export function ReaderSettings() {
  const { look, setLook } = useReader();
  const i = SIZES.indexOf(look.size);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Reading settings"><Settings2Icon /></Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="max-h-[85svh] w-72 overflow-y-auto">
        <DropdownMenuLabel>Text</DropdownMenuLabel>
        <Choice value={look.font} onPick={(font) => setLook({ font })} options={[{ id: "serif", label: "Serif" }, { id: "sans", label: "Sans" }]} />
        <div className="flex items-center justify-between gap-2 px-2 pb-2">
          <Button size="sm" variant="outline" aria-label="Smaller text" disabled={i <= 0} onClick={() => setLook({ size: SIZES[Math.max(0, i - 1)]! })}><MinusIcon /></Button>
          <span className="text-sm tabular-nums">{look.size} px</span>
          <Button size="sm" variant="outline" aria-label="Larger text" disabled={i >= SIZES.length - 1} onClick={() => setLook({ size: SIZES[Math.min(SIZES.length - 1, i + 1)]! })}><PlusIcon /></Button>
        </div>
        <DropdownMenuLabel>Line spacing</DropdownMenuLabel>
        <Choice value={look.spacing} onPick={(spacing) => setLook({ spacing })} options={[{ id: "compact", label: "Compact" }, { id: "normal", label: "Normal" }, { id: "wide", label: "Wide" }]} />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Page</DropdownMenuLabel>
        <Choice value={look.theme} onPick={(theme) => setLook({ theme })} options={[{ id: "light", label: "Light" }, { id: "sepia", label: "Sepia" }, { id: "dark", label: "Dark" }]} />
        <Choice value={look.turn} onPick={(turn) => setLook({ turn })} options={[{ id: "scroll", label: "Scroll" }, { id: "pages", label: "Turn pages" }]} />
        {look.turn === "pages" ? <p className="text-muted-foreground px-2 pb-2 text-xs">Tap the left or right edge, or use the arrow keys or space, to turn a page.</p> : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** The contents: every chapter with its reading time and how far through it the person is. */
export function Contents({ work, chapters, current, suffix = "" }: { work: string; chapters: { title: string; words: number }[]; current: number; /** More of the address to keep, such as the translation ("&t=2199"). */ suffix?: string }) {
  const { at } = useReader();
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm"><ListIcon />Contents</Button>
      </DialogTrigger>
      <DialogContent className="max-h-[85svh] overflow-y-auto sm:max-w-lg">
        <DialogHeader><DialogTitle>Contents</DialogTitle></DialogHeader>
        <ol className="space-y-1">
          {chapters.map((c, k) => {
            const n = k + 1;
            const done = n < current ? 1 : n === current ? at : 0;
            return (
              <li key={n}>
                <Link href={`/learn/works/${work}/read?c=${n}${suffix}`} onClick={() => setOpen(false)} aria-current={n === current ? "page" : undefined} className={cn("hover:bg-muted block rounded-md px-2 py-2", n === current && "bg-muted")}>
                  <span className="flex items-baseline justify-between gap-3 text-sm">
                    <span className="min-w-0 truncate">{c.title}</span>
                    <span className="text-muted-foreground flex shrink-0 items-center gap-1 text-xs tabular-nums">{n < current ? <CheckIcon className="size-3" /> : null}{readingTime(minutesFor(c.words))}</span>
                  </span>
                  <span className="bg-border mt-1.5 block h-1 overflow-hidden rounded-full"><span className="bg-foreground/60 block h-full" style={{ width: `${Math.round(done * 100)}%` }} /></span>
                </Link>
              </li>
            );
          })}
        </ol>
      </DialogContent>
    </Dialog>
  );
}

/** Minutes left in this chapter and in the book, from how far through the chapter the person is. */
export function TimeLeft({ words, chapter }: { words: number[]; chapter: number }) {
  const { at } = useReader();
  const left = timeLeft(words, chapter, at);
  if (!left.book) return <span>Finished</span>;
  return <span>{left.chapter ? `${readingTime(left.chapter)} left in the chapter · ` : ""}{readingTime(left.book)} left in the book</span>;
}
