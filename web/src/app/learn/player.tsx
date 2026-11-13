"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { PauseIcon, PlayIcon, RotateCcwIcon, RotateCwIcon, XIcon } from "lucide-react";
import type { Audiobook } from "@reflect/core/librivox";
import { Button } from "@/components/ui/button";

/** An audiobook playing in Learn: the recording and which book's page it belongs to. */
export type Playing = Audiobook & { work: string; bookTitle: string };
type Place = { chapter: number; at: number };
const keyOf = (id: string) => `reflect:listen:${id}`;
const RATES = [1, 1.25, 1.5, 1.75, 2, 0.75];

/** Where the person stopped in a recording, kept in this browser. */
export function listenPlace(id: string): Place | null {
  try {
    const t = window.localStorage.getItem(keyOf(id));
    return t ? (JSON.parse(t) as Place) : null;
  } catch {
    return null;
  }
}

type Listen = {
  book: Playing | null;
  chapter: number;
  playing: boolean;
  time: number;
  duration: number;
  rate: number;
  /** Whether the reader follows the recording (read and listen together). */
  follow: boolean;
  play: (book: Playing, chapter?: number, opts?: { follow?: boolean }) => void;
  toggle: () => void;
  skip: (seconds: number) => void;
  setFollow: (f: boolean) => void;
  close: () => void;
};
const ListenContext = createContext<Listen | null>(null);
export const useListen = () => useContext(ListenContext);

export const clock = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
};
const stamp = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/**
 * The one audio player for Learn: it keeps playing while the person moves between Learn's pages, goes on to the
 * next part when one ends, and keeps where they stopped in this browser so the book picks up there.
 */
export function ListenProvider({ children }: { children: React.ReactNode }) {
  const audio = useRef<HTMLAudioElement>(null);
  const resume = useRef<number | null>(null);
  const [book, setBook] = useState<Playing | null>(null);
  const [chapter, setChapter] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [follow, setFollow] = useState(false);

  const save = useCallback(() => {
    if (!book || !audio.current) return;
    try {
      window.localStorage.setItem(keyOf(book.id), JSON.stringify({ chapter, at: Math.floor(audio.current.currentTime) } satisfies Place));
    } catch {
      // storage blocked: the player works, it starts from the top next time
    }
  }, [book, chapter]);

  const play = useCallback((b: Playing, c?: number, opts?: { follow?: boolean }) => {
    const p = c === undefined ? listenPlace(b.id) : null;
    const next = c ?? (p && p.chapter < b.chapters.length ? p.chapter : 0);
    resume.current = c === undefined && p && p.chapter === next ? p.at : null;
    setBook(b);
    setChapter(next);
    setFollow(Boolean(opts?.follow));
    setPlaying(true);
    // The same part again: play it from where it is.
    if (book?.id === b.id && chapter === next) void audio.current?.play();
  }, [book, chapter]);

  useEffect(() => {
    if (audio.current) audio.current.playbackRate = rate;
  }, [rate, book, chapter]);

  const value = useMemo<Listen>(() => ({
    book,
    chapter,
    playing,
    time,
    duration,
    rate,
    follow,
    play,
    setFollow,
    toggle: () => {
      const a = audio.current;
      if (!a) return;
      if (a.paused) void a.play();
      else a.pause();
    },
    skip: (s) => {
      const a = audio.current;
      if (a) a.currentTime = Math.max(0, Math.min(a.duration || Infinity, a.currentTime + s));
    },
    close: () => {
      save();
      audio.current?.pause();
      setBook(null);
      setFollow(false);
    },
  }), [book, chapter, playing, time, duration, rate, follow, play, save]);

  const c = book?.chapters[chapter];
  return (
    <ListenContext.Provider value={value}>
      {children}
      {book && c ? (
        <div className="bg-background fixed inset-x-0 bottom-0 z-30 border-t shadow-lg" data-player>
          <audio
            ref={audio}
            key={c.url}
            src={c.url}
            autoPlay
            preload="metadata"
            onLoadedMetadata={(e) => {
              e.currentTarget.playbackRate = rate;
              setDuration(e.currentTarget.duration || c.seconds);
              if (resume.current !== null) {
                e.currentTarget.currentTime = resume.current;
                resume.current = null;
              }
            }}
            onPlay={() => setPlaying(true)}
            onPause={() => { setPlaying(false); save(); }}
            onTimeUpdate={(e) => {
              setTime(e.currentTarget.currentTime);
              if (Math.floor(e.currentTarget.currentTime) % 15 === 0) save();
            }}
            onEnded={() => { if (chapter + 1 < book.chapters.length) setChapter(chapter + 1); else setPlaying(false); }}
          />
          <div className="mx-auto flex max-w-3xl items-center gap-2 px-4 py-2 sm:px-6">
            <Button size="icon" variant="ghost" aria-label="Back 15 seconds" onClick={() => value.skip(-15)}><RotateCcwIcon /></Button>
            <Button size="icon" aria-label={playing ? "Pause" : "Play"} onClick={value.toggle}>{playing ? <PauseIcon /> : <PlayIcon />}</Button>
            <Button size="icon" variant="ghost" aria-label="Forward 30 seconds" onClick={() => value.skip(30)}><RotateCwIcon /></Button>
            <div className="min-w-0 flex-1">
              <Link href={`/learn/works/${book.work}`} className="block truncate text-sm font-medium">{book.bookTitle}</Link>
              <p className="text-muted-foreground truncate text-xs">{c.title} · {stamp(time)} / {stamp(duration || c.seconds)}{follow ? " · the text follows" : ""}</p>
              <div className="bg-border mt-1 h-1 overflow-hidden rounded-full"><div className="bg-foreground/60 h-full" style={{ width: `${duration ? Math.min(100, (time / duration) * 100) : 0}%` }} /></div>
            </div>
            <Button size="sm" variant="ghost" className="tabular-nums" aria-label="Speed" onClick={() => setRate(RATES[(RATES.indexOf(rate) + 1) % RATES.length]!)}>{rate}×</Button>
            <Button size="icon" variant="ghost" aria-label="Close the player" onClick={value.close}><XIcon /></Button>
          </div>
        </div>
      ) : null}
    </ListenContext.Provider>
  );
}
