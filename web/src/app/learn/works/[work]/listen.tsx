"use client";

import { useState } from "react";
import { HeadphonesIcon, PlayIcon, XIcon } from "lucide-react";
import type { Audiobook } from "@reflect/core/librivox";
import { Button } from "@/components/ui/button";
import { clock, useListen } from "../../player";

/**
 * The free audiobook from LibriVox: its parts in order, played by Learn's player, which keeps playing across
 * pages, goes on to the next part, and picks up where you stopped in this browser.
 */
export function Listen({ book, work, title }: { book: Audiobook; work: string; title: string }) {
  const [open, setOpen] = useState(false);
  const listen = useListen();
  const here = listen?.book?.id === book.id;
  const playing = { ...book, work, bookTitle: title };
  if (!open) {
    return (
      <Button variant="outline" onClick={() => { setOpen(true); if (!here) listen?.play(playing); }}>
        <HeadphonesIcon />
        Listen free · {clock(book.seconds)}
      </Button>
    );
  }
  return (
    <section className="space-y-3 rounded-md border p-4">
      <div className="flex items-start justify-between gap-2">
        <p className="text-muted-foreground text-xs">{book.chapters.length} parts · <a href={book.url} target="_blank" rel="noreferrer" className="underline underline-offset-2">LibriVox</a>, read by volunteers</p>
        <Button variant="ghost" size="sm" onClick={() => setOpen(false)}><XIcon />Hide parts</Button>
      </div>
      <ol className="max-h-64 divide-y overflow-y-auto rounded-md border text-sm">
        {book.chapters.map((x, i) => {
          const on = here && listen?.chapter === i;
          return (
            <li key={x.url}>
              <button type="button" onClick={() => listen?.play(playing, i)} className={`hover:bg-muted flex w-full items-center justify-between gap-3 px-3 py-2 text-left ${on ? "bg-muted font-medium" : ""}`} aria-current={on ? "true" : undefined}>
                <span className="flex min-w-0 items-center gap-2">{on ? <PlayIcon className="size-3 shrink-0" /> : null}<span className="truncate">{x.title}</span></span>
                <span className="text-muted-foreground shrink-0 text-xs">{x.seconds ? clock(x.seconds) : ""}{x.reader ? ` · ${x.reader}` : ""}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
