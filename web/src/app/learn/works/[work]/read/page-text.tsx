"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { CheckIcon, CopyIcon, HighlighterIcon, QuoteIcon, Trash2Icon, XIcon } from "lucide-react";
import { STANCES } from "@reflect/core/claims";
import { citeQuote, runsOf, type Span } from "@reflect/core/excerpts";
import { Button } from "@/components/ui/button";
import { addExcerptAction, keepClaimAction, keepPlaceAction, removeExcerptAction } from "../../../actions";

/** A highlight kept in this chapter: its id, its words, and where they sit. */
export type Mark = Span & { id: string; quote: string };

/** Where the person is in a book: kept in this browser, and with their account when the book is on their shelves. */
export type ReadPlace = { chapter: number; chapters: number; at: number };
export const placeKey = (work: string) => `reflect:read:${work}`;

/**
 * One chapter's text. Scrolling keeps the place. Selecting a passage offers to highlight it (kept as an excerpt
 * with its place in the text), to copy it as a quote cited to the author, book and chapter, or to keep it as a
 * claim under the stance the person picks. Tapping a highlight offers to copy or remove it.
 */
export function Page({ work, chapter, chapters, title, author, chapterTitle, paragraphs, bookId, kept: keptPlace, source, marks: initial }: { work: string; chapter: number; chapters: number; title: string; author: string; chapterTitle: string; paragraphs: string[]; /** The book on the person's shelves, when it is there. */ bookId?: string; /** The place kept with their account. */ kept?: ReadPlace | null; /** Where the text comes from, as a highlight records it ("gutenberg:1342"). */ source: string; /** The person's highlights in this chapter. */ marks: Mark[] }) {
  const box = useRef<HTMLDivElement>(null);
  const [picked, setPicked] = useState<{ text: string; span: Span | null; id: string | null } | null>(null);
  const [kept, setKept] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [marks, setMarks] = useState<Mark[]>(initial);
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
    // Where a point in the text is: the paragraph it is in and how many characters into that paragraph.
    const pointOf = (node: Node, offset: number): [number, number] | null => {
      const el = (node.nodeType === Node.ELEMENT_NODE ? (node as Element) : node.parentElement)?.closest("p[data-i]");
      if (!el || !box.current?.contains(el)) return null;
      const r = document.createRange();
      r.selectNodeContents(el);
      r.setEnd(node, offset);
      return [Number(el.getAttribute("data-i")), r.toString().length];
    };
    const onUp = () => {
      const s = window.getSelection();
      const text = s?.toString().replace(/\s+/g, " ").trim() ?? "";
      if (!s || s.rangeCount === 0 || text.length < 3 || !s.anchorNode || !box.current?.contains(s.anchorNode)) return;
      const r = s.getRangeAt(0);
      const a = pointOf(r.startContainer, r.startOffset);
      const b = pointOf(r.endContainer, r.endOffset);
      const span = a && b && (b[0] > a[0] || (b[0] === a[0] && b[1] > a[1])) ? { from: a, to: b } : null;
      setPicked({ text: text.slice(0, 5000), span, id: null });
      setKept(null);
      setDone(null);
      setError(false);
    };
    document.addEventListener("mouseup", onUp);
    document.addEventListener("touchend", onUp);
    return () => {
      document.removeEventListener("mouseup", onUp);
      document.removeEventListener("touchend", onUp);
    };
  }, []);

  const cite = (quote: string) => citeQuote(quote, { author, title, place: chapterTitle });
  const close = () => {
    setPicked(null);
    window.getSelection()?.removeAllRanges();
  };

  const keep = (stance: string) => run(async () => {
    if (!picked) return;
    const f = new FormData();
    f.set("claim", picked.text.slice(0, 1000));
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

  const highlight = () => run(async () => {
    if (!picked?.span) return;
    try {
      const id = await addExcerptAction({ work, quote: picked.text, anchor: { source, chapter, chapterTitle, from: picked.span.from, to: picked.span.to } });
      setMarks((m) => [...m, { id, quote: picked.text, ...picked.span! }]);
      setPicked({ ...picked, id });
      setDone("Highlighted. It is with your excerpts on the book page.");
      window.getSelection()?.removeAllRanges();
    } catch {
      setError(true);
    }
  });

  const remove = () => run(async () => {
    if (!picked?.id) return;
    try {
      await removeExcerptAction(picked.id, work);
      setMarks((m) => m.filter((x) => x.id !== picked.id));
      setPicked(null);
    } catch {
      setError(true);
    }
  });

  const copy = async () => {
    if (!picked) return;
    try {
      await navigator.clipboard.writeText(cite(picked.text));
      setDone("Copied, with the author, book and chapter.");
    } catch {
      setError(true);
    }
  };

  const open = (m: Mark) => {
    if (window.getSelection()?.toString().trim()) return;
    setPicked({ text: m.quote, span: { from: m.from, to: m.to }, id: m.id });
    setKept(null);
    setDone(null);
    setError(false);
  };

  return (
    <>
      <div ref={box} className="writing mt-10 space-y-5 text-pretty">
        {paragraphs.map((p, i) => (
          <p key={i} data-i={i}>
            {runsOf(p, i, marks).map((r, k) => {
              const m = r.id ? marks.find((x) => x.id === r.id) : undefined;
              return m ? <mark key={k} className="cursor-pointer rounded-sm bg-amber-200/70 text-inherit dark:bg-amber-400/30" onClick={() => open(m)}>{r.text}</mark> : <span key={k}>{r.text}</span>;
            })}
          </p>
        ))}
      </div>
      {picked ? (
        <div className="bg-background fixed inset-x-0 bottom-0 z-20 border-t shadow-lg">
          <div className="mx-auto flex max-w-3xl flex-col gap-2 px-4 py-3 sm:px-6">
            <div className="flex items-start justify-between gap-3">
              <p className="text-muted-foreground line-clamp-2 text-sm"><QuoteIcon className="mr-1 inline size-3.5" />{picked.text}</p>
              <Button variant="ghost" size="icon" className="size-7 shrink-0" onClick={close} aria-label="Close"><XIcon /></Button>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {picked.id ? null : <Button size="sm" disabled={pending || !picked.span} onClick={highlight}><HighlighterIcon />Highlight</Button>}
              <Button size="sm" variant="outline" onClick={copy}><CopyIcon />Copy quote</Button>
              {picked.id ? <Button size="sm" variant="ghost" disabled={pending} onClick={remove}><Trash2Icon />Remove highlight</Button> : null}
              {done ? <span className="text-muted-foreground flex items-center gap-1 text-xs"><CheckIcon className="size-3.5" />{done}</span> : null}
            </div>
            {kept ? (
              <p className="flex items-center gap-1.5 text-sm"><CheckIcon className="size-4" />Kept as a claim: {kept}. It is in Claims, cited to {title}.</p>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm">Keep as a claim:</span>
                {STANCES.map((s) => <Button key={s.id} size="sm" variant="outline" disabled={pending} onClick={() => keep(s.id)}>{s.text}</Button>)}
              </div>
            )}
            {error ? <span className="text-destructive text-xs">That did not work. Try again.</span> : null}
          </div>
        </div>
      ) : null}
    </>
  );
}

