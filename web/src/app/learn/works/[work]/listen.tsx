"use client";

import { useEffect, useRef, useState } from "react";
import { HeadphonesIcon, XIcon } from "lucide-react";
import type { Audiobook } from "@reflect/core/librivox";
import { Button } from "@/components/ui/button";

const clock = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
};
type Place = { chapter: number; at: number };
const keyOf = (id: string) => `reflect:listen:${id}`;
function readPlace(id: string): Place | null {
  try {
    const t = window.localStorage.getItem(keyOf(id));
    return t ? (JSON.parse(t) as Place) : null;
  } catch {
    return null;
  }
}

/**
 * The free audiobook from LibriVox, played on the page: its chapters in order, the next one starting when one
 * ends. Where you stopped is kept in this browser, so the book picks up there.
 */
export function Listen({ book }: { book: Audiobook }) {
  const [open, setOpen] = useState(false);
  const [chapter, setChapter] = useState(0);
  const audio = useRef<HTMLAudioElement>(null);
  const resume = useRef<number | null>(null);

  useEffect(() => {
    const p = readPlace(book.id);
    if (p && p.chapter < book.chapters.length) {
      setChapter(p.chapter);
      resume.current = p.at;
    }
  }, [book.id, book.chapters.length]);

  const save = () => {
    try {
      window.localStorage.setItem(keyOf(book.id), JSON.stringify({ chapter, at: Math.floor(audio.current?.currentTime ?? 0) } satisfies Place));
    } catch {
      // storage blocked: the player works, it just starts from the top next time
    }
  };
  const c = book.chapters[chapter]!;

  if (!open) {
    return (
      <Button variant="outline" onClick={() => setOpen(true)}>
        <HeadphonesIcon />
        Listen free · {clock(book.seconds)}
      </Button>
    );
  }
  return (
    <section className="space-y-3 rounded-md border p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{c.title}</p>
          <p className="text-muted-foreground text-xs">Part {chapter + 1} of {book.chapters.length}{c.reader ? ` · read by ${c.reader}` : ""} · <a href={book.url} target="_blank" rel="noreferrer" className="underline underline-offset-2">LibriVox</a>, read by volunteers</p>
        </div>
        <Button variant="ghost" size="sm" onClick={() => { save(); setOpen(false); }}><XIcon />Close</Button>
      </div>
      <audio
        ref={audio}
        key={c.url}
        src={c.url}
        controls
        autoPlay={resume.current === null}
        preload="metadata"
        className="w-full"
        onLoadedMetadata={(e) => {
          if (resume.current !== null) {
            e.currentTarget.currentTime = resume.current;
            resume.current = null;
          }
        }}
        onPause={save}
        onTimeUpdate={(e) => { if (Math.floor(e.currentTarget.currentTime) % 15 === 0) save(); }}
        onEnded={() => { if (chapter + 1 < book.chapters.length) setChapter(chapter + 1); }}
      />
      <ol className="max-h-64 divide-y overflow-y-auto rounded-md border text-sm">
        {book.chapters.map((x, i) => (
          <li key={x.url}>
            <button type="button" onClick={() => { resume.current = null; setChapter(i); }} className={`hover:bg-muted flex w-full justify-between gap-3 px-3 py-2 text-left ${i === chapter ? "bg-muted font-medium" : ""}`} aria-current={i === chapter ? "true" : undefined}>
              <span className="min-w-0 truncate">{x.title}</span>
              <span className="text-muted-foreground shrink-0 text-xs">{x.seconds ? clock(x.seconds) : ""}</span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
