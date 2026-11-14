"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { CheckIcon, CopyIcon, XIcon } from "lucide-react";
import { passageAt, type Passage } from "@reflect/core/courses";
import { clock } from "@reflect/core/media";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { addItemNoteAction, keepUnitPlaceAction, removeItemNoteAction, setUnitDoneAction } from "../actions";
import { loadYouTube, localDay, type YTPlayer } from "./youtube";

type Note = { id: string; at: number | null; quote: string | null; note: string | null };
export type UnitProps = {
  item: string; unit: string; title: string; by: string | null; cite: string;
  youtube: string | null; media: { src: string; kind: "audio" | "video" } | null;
  place: number | null; done: boolean; of: number | null;
  transcript: Passage[]; notes: Note[]; next: string | null;
};

/**
 * One lecture or episode played inside Reflect: the publisher's video through YouTube's player, or its audio or
 * video file in the page's own player. The place is kept as it plays; the transcript sits beside it, follows the
 * speaker, and any timed part seeks the player. Notes are taken at the current moment, and finishing it moves the
 * course or show forward.
 */
export function UnitPlayer(p: UnitProps) {
  const holder = useRef<HTMLDivElement>(null);
  const yt = useRef<YTPlayer | null>(null);
  const el = useRef<HTMLMediaElement | null>(null);
  const list = useRef<HTMLOListElement>(null);
  const [now, setNow] = useState(p.place ?? 0);
  const [follow, setFollow] = useState(true);
  const [q, setQ] = useState("");
  const [text, setText] = useState("");
  const [copied, setCopied] = useState<number | null>(null);
  const [pending, run] = useTransition();
  const router = useRouter();

  const time = useCallback(() => (yt.current ? yt.current.getCurrentTime() : el.current ? el.current.currentTime : 0), []);
  const keep = useCallback(() => { const s = time(); if (s > 1) keepUnitPlaceAction(p.item, p.unit, s, localDay()).catch(() => {}); }, [p.item, p.unit, time]);

  useEffect(() => {
    if (!p.youtube || !holder.current) return;
    let live = true;
    loadYouTube().then((YT) => {
      if (!live || !holder.current) return;
      yt.current = new YT.Player(holder.current, {
        videoId: p.youtube,
        playerVars: { start: Math.floor(p.place ?? 0), rel: 0, modestbranding: 1 },
        events: { onStateChange: (e: { data: number }) => { if (e.data === YT.PlayerState.PAUSED || e.data === YT.PlayerState.ENDED) keep(); } },
      });
    });
    return () => { live = false; keep(); yt.current?.destroy(); yt.current = null; };
  }, [p.youtube, p.place, keep]);

  useEffect(() => {
    const tick = setInterval(() => setNow(time()), 1000);
    const save = setInterval(keep, 15000);
    return () => { clearInterval(tick); clearInterval(save); };
  }, [time, keep]);

  const current = useMemo(() => passageAt(p.transcript, now), [p.transcript, now]);
  useEffect(() => {
    if (!follow || current < 0 || q) return;
    const li = list.current?.querySelector<HTMLElement>(`[data-i="${current}"]`);
    if (li && list.current) list.current.scrollTo({ top: li.offsetTop - list.current.offsetTop - 8, behavior: "smooth" });
  }, [current, follow, q]);

  const seek = (s: number) => {
    if (yt.current) { yt.current.seekTo(s, true); yt.current.playVideo(); }
    else if (el.current) { el.current.currentTime = s; el.current.play().catch(() => {}); }
    setNow(s);
  };
  const hits = useMemo(() => {
    const w = q.trim().toLowerCase();
    return w ? p.transcript.map((x, i) => ({ x, i })).filter(({ x }) => x.text.toLowerCase().includes(w)) : null;
  }, [q, p.transcript]);
  // A passage without its own time takes the time of the chapter it falls under.
  const timeOf = (i: number) => { for (let j = i; j >= 0; j--) if (p.transcript[j]!.at !== null) return p.transcript[j]!.at!; return null; };
  const quoteOf = (x: Passage, i: number) => { const t = timeOf(i); return `“${x.text}” (${p.cite}${t !== null ? `, at ${clock(t)}` : ""})`; };
  const note = (n: { at: number | null; quote?: string; note?: string }) => run(async () => { await addItemNoteAction(p.item, { ...n, unit: p.unit }); setText(""); router.refresh(); });
  const finish = (done: boolean) => run(async () => { keep(); await setUnitDoneAction(p.item, p.unit, done, p.of, localDay()); if (done && p.next) router.push(p.next); else router.refresh(); });
  const shown = hits ?? p.transcript.map((x, i) => ({ x, i }));

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div className="min-w-0 space-y-4">
        {p.youtube ? (
          <div className="aspect-video w-full overflow-hidden rounded-lg border bg-black"><div ref={holder} className="size-full" /></div>
        ) : p.media?.kind === "video" ? (
          <video ref={(v) => { el.current = v; }} src={p.media.src} controls preload="metadata" className="aspect-video w-full rounded-lg border bg-black" onLoadedMetadata={(e) => { if (p.place) e.currentTarget.currentTime = p.place; }} onPause={keep} />
        ) : p.media ? (
          <audio ref={(a) => { el.current = a; }} src={p.media.src} controls preload="metadata" className="w-full" onLoadedMetadata={(e) => { if (p.place) e.currentTarget.currentTime = p.place; }} onPause={keep} />
        ) : <p className="text-muted-foreground text-sm">This one has no recording Reflect can play.</p>}
        {p.place ? <p className="text-muted-foreground text-xs">Picks up at {clock(p.place)}, where you stopped.</p> : null}

        <div className="flex flex-wrap items-center gap-2">
          {p.done ? (
            <>
              <span className="flex items-center gap-1 text-sm"><CheckIcon className="size-4" /> Finished</span>
              <Button size="sm" variant="ghost" disabled={pending} onClick={() => finish(false)}>Not finished</Button>
              {p.next ? <Button size="sm" variant="outline" asChild><Link href={p.next}>Next</Link></Button> : null}
            </>
          ) : <Button size="sm" disabled={pending} onClick={() => finish(true)}>{p.next ? "Finished, next" : "Finished"}</Button>}
        </div>

        <div className="space-y-2">
          <p className="text-sm font-medium">Notes</p>
          {p.notes.length ? (
            <ul className="space-y-2">
              {p.notes.map((n) => (
                <li key={n.id} className="flex items-start gap-2 rounded-md border p-2 text-sm">
                  {n.at !== null ? <button type="button" onClick={() => seek(n.at!)} className="text-muted-foreground shrink-0 tabular-nums underline-offset-4 hover:underline">{clock(n.at)}</button> : null}
                  <span className="min-w-0 flex-1 space-y-1">
                    {n.quote ? <span className="block italic">“{n.quote}”</span> : null}
                    {n.note ? <span className="block">{n.note}</span> : null}
                  </span>
                  <Button type="button" size="icon" variant="ghost" aria-label="Remove note" disabled={pending} onClick={() => run(async () => { await removeItemNoteAction(p.item, n.id); router.refresh(); })}><XIcon /></Button>
                </li>
              ))}
            </ul>
          ) : null}
          <Textarea aria-label="Note" rows={2} maxLength={4000} value={text} onChange={(e) => setText(e.target.value)} placeholder="A note" />
          <Button type="button" size="sm" disabled={pending || !text.trim()} onClick={() => note({ at: Math.floor(time()), note: text })}>Note at {clock(Math.floor(now))}</Button>
        </div>
      </div>

      <div className="min-w-0 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium">Transcript</p>
          {p.transcript.length ? <label className="text-muted-foreground flex items-center gap-1.5 text-xs"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow along</label> : null}
        </div>
        {p.transcript.length ? (
          <>
            <Input aria-label="Search the transcript" placeholder="Search the transcript" value={q} onChange={(e) => setQ(e.target.value)} className="h-9" />
            {hits && !hits.length ? <p className="text-muted-foreground text-sm">Not found.</p> : null}
            <ol ref={list} className="max-h-[70vh] space-y-2 overflow-y-auto rounded-lg border p-3 text-sm leading-relaxed">
              {shown.map(({ x, i }) => {
                const t = timeOf(i);
                return (
                  <li key={i} data-i={i} className={`group rounded-md px-2 py-1 ${i === current ? "bg-muted" : ""}`}>
                    {x.heading ? (
                      <button type="button" disabled={x.at === null} onClick={() => x.at !== null && seek(x.at)} className="text-left font-medium underline-offset-4 hover:underline">
                        {x.text}{x.at !== null ? <span className="text-muted-foreground ml-2 font-normal tabular-nums">{clock(x.at)}</span> : null}
                      </button>
                    ) : (
                      <p>
                        {x.at !== null ? <button type="button" onClick={() => seek(x.at!)} className="text-muted-foreground mr-2 tabular-nums underline-offset-4 hover:underline">{clock(x.at)}</button> : null}
                        {x.who ? <span className="font-medium">{x.who}: </span> : null}{x.text}
                      </p>
                    )}
                    {!x.heading ? (
                      <span className="mt-1 hidden gap-1 group-hover:flex group-focus-within:flex">
                        <Button type="button" size="sm" variant="outline" className="h-7" disabled={pending} onClick={() => note({ at: t, quote: x.text })}>Quote in a note</Button>
                        <Button type="button" size="icon" variant="ghost" className="size-7" aria-label="Copy as a cited quote" onClick={() => { navigator.clipboard?.writeText(quoteOf(x, i)).catch(() => {}); setCopied(i); }}><CopyIcon /></Button>
                        {copied === i ? <span role="status" className="text-muted-foreground self-center text-xs">Copied</span> : null}
                      </span>
                    ) : null}
                  </li>
                );
              })}
            </ol>
          </>
        ) : <p className="text-muted-foreground text-sm">The publisher gives no transcript for this one.</p>}
      </div>
    </div>
  );
}
