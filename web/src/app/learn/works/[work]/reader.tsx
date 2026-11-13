"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { BookOpenIcon, ExternalLinkIcon, XIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

/** Google's preview of an in-copyright book: pages, shown only when Reflect has no text for the book. */
export type Preview = { src: string; full: boolean };
/** A place the book can be found off the page: a scan at the Internet Archive, a loan from Open Library. */
export type ReadLink = { name: string; href: string };

/**
 * Reading the book. When Reflect has the book's text, it is read in Reflect, and the button continues where the
 * person stopped. Without a text, a preview from Google Books opens on the page when there is one, and the places
 * off the page sit beside it.
 */
export function Reader({ here, work, kept, preview, links }: { /** Reflect's own reader for the book, when it has its text. */ here?: string; work: string; /** The place kept with the person's account, when the book is on their shelves. */ kept?: { chapter: number; chapters: number } | null; preview?: Preview | null; links: ReadLink[] }) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<{ chapter: number; chapters: number } | null>(kept ?? null);
  useEffect(() => {
    if (kept || !here) return;
    try {
      const p = JSON.parse(window.localStorage.getItem(`reflect:read:${work}`) ?? "null") as { chapter?: number; chapters?: number } | null;
      if (p?.chapter && p.chapters) setPlace({ chapter: p.chapter, chapters: p.chapters });
    } catch {
      // nothing kept
    }
  }, [work, kept, here]);
  const pages = !here && preview ? preview : null;
  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {here ? (
          <Button asChild>
            <Link href={place ? `${here}?c=${place.chapter}` : here}><BookOpenIcon />{place ? `Continue reading · ${place.chapter} of ${place.chapters}` : "Read it free"}</Link>
          </Button>
        ) : null}
        {pages && !open ? <Button variant="outline" onClick={() => setOpen(true)}><BookOpenIcon />{pages.full ? "Pages from Google Books" : "Preview pages from Google Books"}</Button> : null}
        {links.map((l) => <Button key={l.href} asChild variant="ghost" size="sm"><a href={l.href} target="_blank" rel="noreferrer">{l.name}<ExternalLinkIcon /></a></Button>)}
      </div>
      {here ? <p className="text-muted-foreground text-xs">The book&apos;s text, read in Reflect. Highlight passages and keep excerpts as you read.</p> : null}
      {pages && open ? (
        <>
          <div className="flex items-center justify-between gap-2">
            <p className="text-muted-foreground text-xs">Page images from Google Books. Reflect has no text for this book, so it cannot be highlighted.</p>
            <Button variant="ghost" size="sm" onClick={() => setOpen(false)}><XIcon />Close</Button>
          </div>
          <iframe src={pages.src} title="Pages from Google Books" className="bg-muted h-[75vh] min-h-96 w-full rounded-md border" allowFullScreen />
        </>
      ) : null}
    </section>
  );
}
