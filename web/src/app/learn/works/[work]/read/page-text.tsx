"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { CheckIcon, QuoteIcon, XIcon } from "lucide-react";
import { STANCES } from "@reflect/core/claims";
import { Button } from "@/components/ui/button";
import { keepClaimAction, keepPlaceAction } from "../../../actions";

/** Where the person is in a book: kept in this browser, and with their account when the book is on their shelves. */
export type ReadPlace = { chapter: number; chapters: number; at: number };
export const placeKey = (work: string) => `reflect:read:${work}`;

/**
 * One chapter's text. Scrolling keeps the place; selecting a passage offers to keep it as a claim, with the
 * author as its holder and the book and chapter as its citation, under the stance the person picks.
 */
export function Page({ work, chapter, chapters, title, author, chapterTitle, paragraphs, bookId, kept: keptPlace }: { work: string; chapter: number; chapters: number; title: string; author: string; chapterTitle: string; paragraphs: string[]; /** The book on the person's shelves, when it is there. */ bookId?: string; /** The place kept with their account. */ kept?: ReadPlace | null }) {
  const box = useRef<HTMLDivElement>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [kept, setKept] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const [pending, run] = useTransition();

  useEffect(() => {
    let saved: ReadPlace | null = null;
    try {
      saved = JSON.parse(window.localStorage.getItem(placeKey(work)) ?? "null") as ReadPlace | null;
    } catch {
      saved = null;
    }
    // The place kept with the account wins: it may come from another device.
    if (keptPlace) saved = keptPlace;
    if (saved?.chapter === chapter && saved.at > 0) {
      requestAnimationFrame(() => window.scrollTo(0, saved!.at * (document.documentElement.scrollHeight - window.innerHeight)));
    } else {
      window.scrollTo(0, 0);
    }
    let t: ReturnType<typeof setTimeout> | undefined;
    let remote: ReturnType<typeof setTimeout> | undefined;
    const here = (): ReadPlace => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      return { chapter, chapters, at: h > 0 ? Math.min(1, window.scrollY / h) : 0 };
    };
    const send = () => {
      if (bookId) keepPlaceAction(bookId, here()).catch(() => undefined);
    };
    const save = () => {
      try {
        window.localStorage.setItem(placeKey(work), JSON.stringify(here()));
      } catch {
        // storage blocked: reading works, the place is not kept in this browser
      }
    };
    const onScroll = () => {
      clearTimeout(t);
      t = setTimeout(save, 300);
      // The account is told less often: once the person has stopped scrolling for a few seconds.
      clearTimeout(remote);
      remote = setTimeout(send, 4000);
    };
    const onHide = () => {
      if (document.visibilityState === "hidden") send();
    };
    save();
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("visibilitychange", onHide);
    return () => {
      clearTimeout(t);
      clearTimeout(remote);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("visibilitychange", onHide);
    };
  }, [work, chapter, chapters, bookId, keptPlace]);

  useEffect(() => {
    const onUp = () => {
      const s = window.getSelection();
      const text = s?.toString().replace(/\s+/g, " ").trim() ?? "";
      if (text.length >= 12 && s?.anchorNode && box.current?.contains(s.anchorNode)) {
        setPicked(text.slice(0, 1000));
        setKept(null);
        setError(false);
      }
    };
    document.addEventListener("mouseup", onUp);
    document.addEventListener("touchend", onUp);
    return () => {
      document.removeEventListener("mouseup", onUp);
      document.removeEventListener("touchend", onUp);
    };
  }, []);

  const keep = (stance: string) => run(async () => {
    if (!picked) return;
    const f = new FormData();
    f.set("claim", picked);
    f.set("holders", `${author || title}|${title}, ${chapterTitle}`);
    f.set("stance", stance);
    try {
      await keepClaimAction(f);
      setKept(STANCES.find((x) => x.id === stance)?.text ?? stance);
      window.getSelection()?.removeAllRanges();
    } catch {
      setError(true);
    }
  });

  return (
    <>
      <div ref={box} className="writing mt-10 space-y-5 text-pretty">
        {paragraphs.map((p, i) => <p key={i}>{p}</p>)}
      </div>
      {picked ? (
        <div className="bg-background fixed inset-x-0 bottom-0 z-20 border-t shadow-lg">
          <div className="mx-auto flex max-w-3xl flex-col gap-2 px-4 py-3 sm:px-6">
            <div className="flex items-start justify-between gap-3">
              <p className="text-muted-foreground line-clamp-2 text-sm"><QuoteIcon className="mr-1 inline size-3.5" />{picked}</p>
              <Button variant="ghost" size="icon" className="size-7 shrink-0" onClick={() => setPicked(null)} aria-label="Close"><XIcon /></Button>
            </div>
            {kept ? (
              <p className="flex items-center gap-1.5 text-sm"><CheckIcon className="size-4" />Kept as a claim: {kept}. It is in Claims, cited to {title}.</p>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm">Keep as a claim:</span>
                {STANCES.map((s) => <Button key={s.id} size="sm" variant="outline" disabled={pending} onClick={() => keep(s.id)}>{s.text}</Button>)}
                {error ? <span className="text-destructive text-xs">That did not save. Try again.</span> : null}
              </div>
            )}
          </div>
        </div>
      ) : null}
    </>
  );
}
