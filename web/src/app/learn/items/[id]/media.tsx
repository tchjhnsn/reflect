"use client";

import { useEffect, useMemo, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { CopyIcon, XIcon } from "lucide-react";
import { clock, readTranscript, searchTranscript, type Embed } from "@reflect/core/media";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { addItemNoteAction, keepItemPlaceAction, removeItemNoteAction } from "../../actions";

type Note = { id: string; at: number | null; quote: string | null; note: string | null };
type YTPlayer = { getCurrentTime(): number; seekTo(s: number, allow: boolean): void; playVideo(): void; destroy(): void };
type YTNamespace = { Player: new (el: HTMLElement, o: Record<string, unknown>) => YTPlayer; PlayerState: { PAUSED: number; ENDED: number } };
declare global {
  interface Window { YT?: YTNamespace; onYouTubeIframeAPIReady?: () => void }
}

const localDay = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };

function loadYouTube(): Promise<YTNamespace> {
  return new Promise((resolve) => {
    if (window.YT?.Player) return resolve(window.YT);
    const prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => { prev?.(); resolve(window.YT!); };
    if (!document.querySelector('script[src="https://www.youtube.com/iframe_api"]')) {
      const s = document.createElement("script");
      s.src = "https://www.youtube.com/iframe_api";
      document.head.appendChild(s);
    }
  });
}

/**
 * Watching or listening inside Reflect. A YouTube video plays through YouTube's player API: the place is kept as
 * it plays (every 15 seconds and on pause), notes are taken at the current moment, and a note's time seeks there.
 * Other sites play in their own embed, where notes take a time typed by hand. A pasted transcript is searched here
 * and any part quoted into a note with its moment, or copied as a cited quote.
 */
export function ItemMedia({ id, title, by, embed, place, notes, transcript }: { id: string; title: string; by: string | null; embed: Embed | null; place: number | null; notes: Note[]; transcript: string | null }) {
  const holder = useRef<HTMLDivElement>(null);
  const player = useRef<YTPlayer | null>(null);
  const [text, setText] = useState("");
  const [at, setAt] = useState("");
  const [q, setQ] = useState("");
  const [copied, setCopied] = useState<number | null>(null);
  const [pending, run] = useTransition();
  const router = useRouter();
  const parts = useMemo(() => (transcript ? readTranscript(transcript) : []), [transcript]);
  const hits = useMemo(() => searchTranscript(parts, q), [parts, q]);

  useEffect(() => {
    if (embed?.kind !== "youtube" || !holder.current) return;
    let live = true;
    let timer: ReturnType<typeof setInterval> | undefined;
    const keep = () => { const s = player.current?.getCurrentTime(); if (s && s > 1) keepItemPlaceAction(id, s, localDay()).catch(() => {}); };
    loadYouTube().then((YT) => {
      if (!live || !holder.current) return;
      player.current = new YT.Player(holder.current, {
        videoId: embed.id,
        playerVars: { start: Math.floor(place ?? embed.start ?? 0), rel: 0, modestbranding: 1 },
        events: { onStateChange: (e: { data: number }) => { if (e.data === YT.PlayerState.PAUSED || e.data === YT.PlayerState.ENDED) keep(); } },
      });
      timer = setInterval(keep, 15000);
    });
    return () => { live = false; if (timer) clearInterval(timer); keep(); player.current?.destroy(); player.current = null; };
  }, [embed, id, place]);

  const now = () => (player.current ? Math.floor(player.current.getCurrentTime()) : null);
  const seek = (s: number) => { player.current?.seekTo(s, true); player.current?.playVideo(); holder.current?.scrollIntoView?.({ block: "nearest" }); };
  const typed = () => { const m = /^(?:(\d+):)?(\d{1,2}):(\d{2})$/.exec(at.trim()); return m ? Number(m[1] ?? 0) * 3600 + Number(m[2]) * 60 + Number(m[3]) : null; };
  const save = (n: { at: number | null; quote?: string; note?: string }) => run(async () => { await addItemNoteAction(id, n); setText(""); setAt(""); router.refresh(); });
  const cite = (p: { at: number; text: string }) => `“${p.text}” (${[by, title].filter(Boolean).join(", ")}, at ${clock(p.at)})`;

  return (
    <div className="space-y-4">
      {embed?.kind === "youtube" ? (
        <div className="aspect-video w-full overflow-hidden rounded-lg border bg-black"><div ref={holder} className="size-full" /></div>
      ) : embed ? (
        <iframe src={embed.src} title={`${title} on ${embed.site}`} className={embed.tall ? "aspect-video w-full rounded-lg border" : "h-40 w-full rounded-lg border"} allow="autoplay; encrypted-media; fullscreen; picture-in-picture" allowFullScreen loading="lazy" />
      ) : null}
      {embed?.kind === "youtube" && place ? <p className="text-muted-foreground text-xs">Picks up at {clock(place)}, where you stopped.</p> : null}

      <div className="space-y-2">
        <p className="text-sm font-medium">Notes</p>
        {notes.length ? (
          <ul className="space-y-2">
            {notes.map((n) => (
              <li key={n.id} className="flex items-start gap-2 rounded-md border p-2 text-sm">
                {n.at !== null ? (
                  embed?.kind === "youtube" ? <button type="button" onClick={() => seek(n.at!)} className="text-muted-foreground shrink-0 tabular-nums underline-offset-4 hover:underline">{clock(n.at)}</button> : <span className="text-muted-foreground shrink-0 tabular-nums">{clock(n.at)}</span>
                ) : null}
                <span className="min-w-0 flex-1 space-y-1">
                  {n.quote ? <span className="block italic">“{n.quote}”</span> : null}
                  {n.note ? <span className="block">{n.note}</span> : null}
                </span>
                <Button type="button" size="icon" variant="ghost" aria-label="Remove note" disabled={pending} onClick={() => run(async () => { await removeItemNoteAction(id, n.id); router.refresh(); })}><XIcon /></Button>
              </li>
            ))}
          </ul>
        ) : <p className="text-muted-foreground text-sm">No notes yet.</p>}
        <div className="space-y-2">
          <Textarea aria-label="Note" rows={2} maxLength={4000} value={text} onChange={(e) => setText(e.target.value)} placeholder="A note" />
          <div className="flex flex-wrap items-center gap-2">
            {embed?.kind === "youtube" ? (
              <Button type="button" size="sm" disabled={pending || !text.trim()} onClick={() => save({ at: now(), note: text })}>Note at this moment</Button>
            ) : (
              <>
                <Input aria-label="At (m:ss)" placeholder="m:ss" value={at} onChange={(e) => setAt(e.target.value)} className="h-8 w-24" />
                <Button type="button" size="sm" disabled={pending || !text.trim()} onClick={() => save({ at: typed(), note: text })}>Keep note</Button>
              </>
            )}
          </div>
        </div>
      </div>

      {parts.length ? (
        <div className="space-y-2">
          <p className="text-sm font-medium">Search the transcript</p>
          <Input aria-label="Search the transcript" placeholder="A word or phrase" value={q} onChange={(e) => setQ(e.target.value)} />
          {q && !hits.length ? <p className="text-muted-foreground text-sm">Not found.</p> : null}
          <ul className="space-y-2">
            {hits.map((p, i) => (
              <li key={`${p.at}-${i}`} className="flex items-start gap-2 rounded-md border p-2 text-sm">
                {embed?.kind === "youtube" ? <button type="button" onClick={() => seek(p.at)} className="text-muted-foreground shrink-0 tabular-nums underline-offset-4 hover:underline">{clock(p.at)}</button> : <span className="text-muted-foreground shrink-0 tabular-nums">{clock(p.at)}</span>}
                <span className="min-w-0 flex-1">{p.text}</span>
                <span className="flex shrink-0 gap-1">
                  <Button type="button" size="sm" variant="outline" disabled={pending} onClick={() => save({ at: p.at, quote: p.text })}>Quote in a note</Button>
                  <Button type="button" size="icon" variant="ghost" aria-label="Copy as a cited quote" onClick={() => { navigator.clipboard?.writeText(cite(p)).catch(() => {}); setCopied(i); }}><CopyIcon /></Button>
                </span>
                {copied === i ? <span className="sr-only" role="status">Copied</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
