"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { CheckIcon, ChevronRightIcon } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { lectureMediaAction } from "../actions";

type Row = { n: number; title: string; href: string; done: boolean; youtube: string | null };
type Media = { youtube: string | null; video: string | null; audio: string | null } | null;

function Player({ courseUrl, row }: { courseUrl: string; row: Row }) {
  const [media, setMedia] = useState<Media | "loading">(row.youtube ? { youtube: row.youtube, video: null, audio: null } : "loading");
  useEffect(() => {
    if (row.youtube) return;
    let live = true;
    lectureMediaAction(courseUrl, row.n).then((m) => { if (live) setMedia(m); }).catch(() => { if (live) setMedia(null); });
    return () => { live = false; };
  }, [courseUrl, row.n, row.youtube]);
  if (media === "loading") {
    return <div className="bg-muted aspect-video w-full animate-pulse rounded-lg" aria-label="Loading the lecture" />;
  }
  if (media?.youtube) return <iframe src={`https://www.youtube-nocookie.com/embed/${media.youtube}?rel=0`} title={row.title} allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen className="aspect-video w-full rounded-lg border bg-black" />;
  if (media?.video) return <video src={media.video} controls preload="metadata" className="aspect-video w-full rounded-lg border bg-black" />;
  if (media?.audio) return <audio src={media.audio} controls preload="metadata" className="w-full" />;
  return <p className="text-muted-foreground text-sm">This one did not load. Open the lecture to try again.</p>;
}

/**
 * A course's lectures, each opening in place to play its recording without leaving the list; the lecture's own page
 * has the transcript and, once the course is kept, notes and your place.
 */
export function LectureList({ courseUrl, rows }: { courseUrl: string; rows: Row[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <ol className="divide-y rounded-lg border">
      {rows.map((r) => (
        <Collapsible key={r.n} asChild open={open === r.n} onOpenChange={(o) => setOpen(o ? r.n : null)}>
          <li>
            <div className="flex items-center gap-2 px-3 py-2.5 text-sm">
              <CollapsibleTrigger className="hover:bg-muted -m-1 flex min-w-0 flex-1 items-center gap-2 rounded-md p-1 text-left" aria-label={`${open === r.n ? "Close" : "Play"} ${r.title}`}>
                <ChevronRightIcon className={`text-muted-foreground size-4 shrink-0 transition ${open === r.n ? "rotate-90" : ""}`} />
                <span className="text-muted-foreground w-5 shrink-0 text-right tabular-nums">{r.done ? <CheckIcon className="ml-auto size-4" aria-label="Finished" /> : r.n}</span>
                <span className="min-w-0">{r.title}</span>
              </CollapsibleTrigger>
              <Link href={r.href} className="text-muted-foreground hover:text-foreground shrink-0 text-xs underline-offset-4 hover:underline">Open</Link>
            </div>
            <CollapsibleContent className="px-3 pb-3">{open === r.n ? <Player courseUrl={courseUrl} row={r} /> : null}</CollapsibleContent>
          </li>
        </Collapsible>
      ))}
    </ol>
  );
}
