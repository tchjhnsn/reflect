"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { searchMineAction, type Found } from "@/app/learn/search-action";

/**
 * Search from any page: what is on your shelves matches as you type, and Enter searches everything in Browse (books,
 * courses, lectures, podcasts and the rest). Opens with the button, / or Cmd-K (`keys`
 * false for a second copy of the button on the same page).
 */
export function SearchButton({ keys = true }: { keys?: boolean }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [found, setFound] = useState<Found[]>([]);
  const [, run] = useTransition();
  const ask = useRef(0);

  useEffect(() => {
    if (!keys) return;
    const key = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && (e.target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName));
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) {
        e.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [keys]);

  useEffect(() => {
    const n = ++ask.current;
    if (q.trim().length < 2) {
      setFound([]);
      return;
    }
    const t = setTimeout(() => run(async () => {
      const r = await searchMineAction(q);
      if (n === ask.current) setFound(r);
    }), 200);
    return () => clearTimeout(t);
  }, [q]);

  const everything = () => {
    if (!q.trim()) return;
    setOpen(false);
    router.push(`/learn/browse?m=all&q=${encodeURIComponent(q.trim())}`);
  };

  return (
    <>
      <Button variant="ghost" size="icon" aria-label="Search" title="Search (/)" onClick={() => setOpen(true)}><SearchIcon /></Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="top-[20%] translate-y-0 gap-3 p-4 sm:max-w-lg">
          <DialogTitle className="sr-only">Search</DialogTitle>
          <DialogDescription className="sr-only">Search your shelves, or everything in Browse.</DialogDescription>
          <form onSubmit={(e) => { e.preventDefault(); everything(); }}>
            <Input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search books, courses, lectures, podcasts..." aria-label="Search" className="h-10" />
          </form>
          {found.length ? (
            <div className="space-y-1">
              <p className="text-muted-foreground px-1 text-xs font-medium">On your shelves</p>
              <ul>
                {found.map((f) => (
                  <li key={f.href}><Link href={f.href} onClick={() => setOpen(false)} className="hover:bg-muted flex items-baseline justify-between gap-3 rounded-md px-2 py-1.5 text-sm"><span className="min-w-0 truncate">{f.title}</span><span className="text-muted-foreground shrink-0 text-xs">{f.kind}</span></Link></li>
                ))}
              </ul>
            </div>
          ) : null}
          {q.trim() ? (
            <button type="button" onClick={everything} className="hover:bg-muted flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm"><SearchIcon className="size-4" />Search everything for “{q.trim()}”</button>
          ) : <p className="text-muted-foreground px-1 text-xs">Press Enter to search everything in Browse.</p>}
        </DialogContent>
      </Dialog>
    </>
  );
}
