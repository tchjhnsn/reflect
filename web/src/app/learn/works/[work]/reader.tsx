"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { BookOpenIcon, ExternalLinkIcon, XIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

/** A reader the book can be read in on the page: the Internet Archive's or Google's. */
export type ReadSource = { key: string; name: string; src: string; full: boolean };
/** A place the book can be read off the page: Project Gutenberg, Standard Ebooks, a loan from Open Library. */
export type ReadLink = { name: string; href: string };

/**
 * Reading the book: the first reader opens on the page when the person asks, the whole book when one has it
 * free and a preview otherwise; the other readers and the places off the page sit beside it.
 */
export function Reader({ sources, links, here, work, kept }: { sources: ReadSource[]; links: ReadLink[]; /** Reflect's own reader for the book, when it has a text for it. */ here?: string; work: string; /** The place kept with the person's account, when the book is on their shelves. */ kept?: { chapter: number; chapters: number } | null }) {
  const [open, setOpen] = useState<string | null>(null);
  const [place, setPlace] = useState<{ chapter: number; chapters: number } | null>(kept ?? null);
  useEffect(() => {
    if (kept) return;
    try {
      const p = JSON.parse(window.localStorage.getItem(`reflect:read:${work}`) ?? "null") as { chapter?: number; chapters?: number } | null;
      if (p?.chapter && p.chapters) setPlace({ chapter: p.chapter, chapters: p.chapters });
    } catch {
      // nothing kept
    }
  }, [work, kept]);
  const first = here ? undefined : sources[0];
  const current = sources.find((s) => s.key === open);
  const label = first?.full ? "Read it free" : "Read a preview";
  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {here ? (
          <Button asChild variant="outline">
            <Link href={place ? `${here}?c=${place.chapter}` : here}><BookOpenIcon />{place ? `Continue reading · ${place.chapter} of ${place.chapters}` : "Read it free"}</Link>
          </Button>
        ) : null}
        {first && !current ? <Button variant="outline" onClick={() => setOpen(first.key)}><BookOpenIcon />{label}</Button> : null}
        {sources.filter((s) => s.key !== current?.key && s.key !== first?.key).map((s) => <Button key={s.key} variant="ghost" size="sm" onClick={() => setOpen(s.key)}>{s.key === "archive" ? "Scanned book" : s.full ? "Whole book" : "Preview"} at {s.name}</Button>)}
        {links.map((l) => <Button key={l.href} asChild variant="ghost" size="sm"><a href={l.href} target="_blank" rel="noreferrer">{l.name}<ExternalLinkIcon /></a></Button>)}
      </div>
      {current ? (
        <>
          <div className="flex items-center justify-between gap-2">
            <p className="text-muted-foreground text-xs">{current.full ? "The whole book, free" : "A preview"}, from {current.name}. Search inside it with the reader&apos;s own search.</p>
            <Button variant="ghost" size="sm" onClick={() => setOpen(null)}><XIcon />Close</Button>
          </div>
          <iframe key={current.key} src={current.src} title={`${current.full ? "The whole book" : "A preview"}, from ${current.name}`} className="bg-muted h-[75vh] min-h-96 w-full rounded-md border" allowFullScreen />
        </>
      ) : null}
    </section>
  );
}
