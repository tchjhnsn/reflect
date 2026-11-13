"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { BookAIcon, CheckIcon, CopyIcon, HighlighterIcon, MessageSquareIcon, QuoteIcon, Trash2Icon, XIcon } from "lucide-react";
import { STANCES } from "@reflect/core/claims";
import { citeQuote, cutRuns, HIGHLIGHT_COLORS, runsOf, type HighlightColor, type Span } from "@reflect/core/excerpts";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { COLOR } from "../../../colors";
import { lookupWord, type Definition } from "@reflect/core/wiktionary";
import { Button } from "@/components/ui/button";
import { addExcerptAction, defineAction, keepClaimAction, keepPlaceAction, removeExcerptAction, reviseExcerptAction } from "../../../actions";
import { useReader } from "./reader-frame";
import { useListen } from "../../../player";

/** A highlight kept in this chapter: its id, its words, and where they sit. */
export type Mark = Span & { id: string; quote: string; color: HighlightColor; note: string | null };

/** Where the person is in a book: kept in this browser, and with their account when the book is on their shelves. */
export type ReadPlace = { chapter: number; chapters: number; at: number };
export const placeKey = (work: string) => `reflect:read:${work}`;

/**
 * One chapter's text. Scrolling keeps the place. Selecting a passage offers to highlight it (kept as an excerpt
 * with its place in the text), to copy it as a quote cited to the author, book and chapter, or to keep it as a
 * claim under the stance the person picks. Tapping a highlight offers to copy or remove it.
 */
export function Page({ work, chapter, chapters, title, author, chapterTitle, paragraphs, bookId, kept: keptPlace, source, marks: initial, notes = [], noteText = {}, found = [], goTo, next }: { work: string; chapter: number; chapters: number; title: string; author: string; chapterTitle: string; paragraphs: string[]; /** The book on the person's shelves, when it is there. */ bookId?: string; /** The place kept with their account, already carried to this text's chapters. */ kept?: { chapter: number; at: number } | null; /** Where the text comes from, as a highlight records it ("gutenberg:1342"). */ source: string; /** The person's highlights in this chapter. */ marks: Mark[]; /** Note marks in this chapter, and each note's text. */ notes?: { p: number; c: number; id: string }[]; noteText?: Record<string, string>; /** Where a searched phrase occurs in this chapter. */ found?: Span[]; /** The paragraph to open at (a search result), instead of the kept place. */ goTo?: number; /** The next chapter's address, for turning past the last page. */ next?: string }) {
  const box = useRef<HTMLDivElement>(null);
  const [picked, setPicked] = useState<{ text: string; span: Span | null; id: string | null } | null>(null);
  const [kept, setKept] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [marks, setMarks] = useState<Mark[]>(initial);
  const [error, setError] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [writing, setWriting] = useState<string | null>(null);
  const [defined, setDefined] = useState<{ word: string; definitions: Definition[]; lemma?: { word: string; definitions: Definition[] } } | "none" | null>(null);
  const [pending, run] = useTransition();
  const { look, setAt } = useReader();
  // The bars sit above Learn's audio player when it is open.
  const lift = useListen()?.book ? "bottom-[60px]" : "bottom-0";
  const router = useRouter();

  useEffect(() => {
    let saved: ReadPlace | null = null;
    try {
      saved = JSON.parse(window.localStorage.getItem(placeKey(work)) ?? "null") as ReadPlace | null;
    } catch {
      saved = null;
    }
    // The place kept with the account wins: it may come from another device.
    if (keptPlace) saved = { ...keptPlace, chapters };
    const target = goTo !== undefined ? document.querySelector(`p[data-i="${goTo}"]`) : null;
    if (target) {
      requestAnimationFrame(() => target.scrollIntoView({ block: "center" }));
    } else if (saved?.chapter === chapter && saved.at > 0) {
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
      setAt(here().at);
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
  }, [work, chapter, chapters, bookId, keptPlace, goTo, setAt]);

  useEffect(() => {
    if (look.turn !== "pages") return;
    // Turning pages: a screen at a time, by the arrow keys, space, or a tap at the left or right edge; past the
    // last page of the chapter, on to the next.
    const turn = (dir: 1 | -1) => {
      const end = document.documentElement.scrollHeight - window.innerHeight;
      if (dir > 0 && window.scrollY >= end - 2 && next) {
        router.push(next);
        return;
      }
      window.scrollBy({ top: dir * (window.innerHeight - 72), behavior: "smooth" });
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("input, textarea, [contenteditable]")) return;
      if (e.key === "ArrowRight" || e.key === "PageDown" || (e.key === " " && !e.shiftKey)) {
        e.preventDefault();
        turn(1);
      } else if (e.key === "ArrowLeft" || e.key === "PageUp" || (e.key === " " && e.shiftKey)) {
        e.preventDefault();
        turn(-1);
      }
    };
    const onClick = (e: MouseEvent) => {
      if (window.getSelection()?.toString().trim()) return;
      if (e.target instanceof HTMLElement && e.target.closest("a, button, mark, sup, [role=dialog], [data-bar]")) return;
      const edge = window.innerWidth * 0.25;
      if (e.clientX > window.innerWidth - edge) turn(1);
      else if (e.clientX < edge) turn(-1);
    };
    window.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onClick);
    };
  }, [look.turn, next, router]);

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
      setDefined(null);
      setNote(null);
      setWriting(null);
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

  const highlight = (color: HighlightColor = "yellow") => run(async () => {
    if (!picked?.span) return;
    try {
      const id = await addExcerptAction({ work, quote: picked.text, color, anchor: { source, chapter, chapterTitle, from: picked.span.from, to: picked.span.to } });
      setMarks((m) => [...m, { id, quote: picked.text, color, note: null, ...picked.span! }]);
      setPicked({ ...picked, id });
      setDone("Highlighted. It is with your excerpts on the book page.");
      window.getSelection()?.removeAllRanges();
    } catch {
      setError(true);
    }
  });

  // A new colour or note keeps the highlight as a new excerpt with a new id (an excerpt is never edited).
  const revise = (change: { color?: HighlightColor; note?: string }) => run(async () => {
    if (!picked?.id) return;
    try {
      const id = await reviseExcerptAction(picked.id, work, change);
      setMarks((m) => m.map((x) => (x.id === picked.id ? { ...x, id, ...(change.color ? { color: change.color } : {}), ...(change.note !== undefined ? { note: change.note.trim() || null } : {}) } : x)));
      setPicked({ ...picked, id });
      setWriting(null);
      setDone(change.note !== undefined ? "Note kept." : "Colour changed.");
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

  const define = () => run(async () => {
    if (!picked) return;
    try {
      setDefined((await defineAction(picked.text)) ?? "none");
    } catch {
      setError(true);
    }
  });

  const open = (m: Mark) => {
    if (window.getSelection()?.toString().trim()) return;
    setPicked({ text: m.quote, span: { from: m.from, to: m.to }, id: m.id });
    setDefined(null);
    setNote(null);
    setWriting(null);
    setKept(null);
    setDone(null);
    setError(false);
  };

  return (
    <>
      <div ref={box} className="reading mt-10 space-y-[0.9em] text-pretty">
        {paragraphs.map((p, i) => {
          const here = notes.filter((n) => n.p === i);
          const all = [...marks, ...found.map((f) => ({ ...f, id: "found" }))];
          return (
            <p key={i} data-i={i}>
              {cutRuns(runsOf(p, i, all), here.map((n) => n.c)).map((r, k) => {
                const m = r.id && r.id !== "found" ? marks.find((x) => x.id === r.id) : undefined;
                const refs = here.filter((n) => n.c === r.at + r.text.length);
                const body = m ? <mark className={cn("cursor-pointer rounded-sm text-inherit", COLOR[m.color].mark, m.note && "underline decoration-dotted underline-offset-4")} onClick={() => open(m)}>{r.text}</mark> : r.id === "found" ? <mark className="rounded-sm bg-sky-200/80 text-inherit dark:bg-sky-400/30">{r.text}</mark> : <span>{r.text}</span>;
                return (
                  <span key={k}>
                    {body}
                    {refs.map((n) => (
                      <sup key={n.id} className="leading-none"><button type="button" className="text-primary inline-block min-w-4 px-1 py-0.5 text-[0.7em] leading-none font-semibold underline-offset-2 hover:underline" aria-label={`Note ${n.id.replace("note-", "")}`} onClick={() => { setNote(n.id); setPicked(null); }}>{n.id.replace("note-", "")}</button></sup>
                    ))}
                  </span>
                );
              })}
            </p>
          );
        })}
      </div>
      {note ? (
        <div data-bar className={cn("bg-background fixed inset-x-0 z-20 border-t shadow-lg", lift)}>
          <div className="mx-auto flex max-w-3xl items-start justify-between gap-3 px-4 py-3 sm:px-6">
            <p className="max-h-[40svh] overflow-y-auto text-sm"><span className="font-semibold">Note {note.replace("note-", "")}. </span>{noteText[note] ?? "This note is not in the text."}</p>
            <Button variant="ghost" size="icon" className="size-7 shrink-0" onClick={() => setNote(null)} aria-label="Close"><XIcon /></Button>
          </div>
        </div>
      ) : null}
      {picked ? (
        <div data-bar className={cn("bg-background fixed inset-x-0 z-20 border-t shadow-lg", lift)}>
          <div className="mx-auto flex max-h-[70svh] max-w-3xl flex-col gap-2 overflow-y-auto px-4 py-3 sm:px-6">
            <div className="flex items-start justify-between gap-3">
              <p className="text-muted-foreground line-clamp-2 text-sm"><QuoteIcon className="mr-1 inline size-3.5" />{picked.text}</p>
              <Button variant="ghost" size="icon" className="size-7 shrink-0" onClick={close} aria-label="Close"><XIcon /></Button>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {picked.id ? null : <Button size="sm" disabled={pending || !picked.span} onClick={() => highlight()}><HighlighterIcon />Highlight</Button>}
              <span className="inline-flex items-center gap-1.5 px-1" role="group" aria-label={picked.id ? "Colour" : "Highlight in a colour"}>
                {HIGHLIGHT_COLORS.map((c) => {
                  const on = picked.id ? marks.find((x) => x.id === picked.id)?.color === c : false;
                  return <button key={c} type="button" aria-label={COLOR[c].name} aria-pressed={on} disabled={pending || (!picked.id && !picked.span) || on} onClick={() => (picked.id ? revise({ color: c }) : highlight(c))} className={cn("size-5 rounded-full", COLOR[c].dot, on ? "ring-foreground/60 ring-2 ring-offset-1" : "opacity-70 hover:opacity-100")} />;
                })}
              </span>
              {picked.id && writing === null ? <Button size="sm" variant="outline" onClick={() => setWriting(marks.find((x) => x.id === picked.id)?.note ?? "")}><MessageSquareIcon />{marks.find((x) => x.id === picked.id)?.note ? "Edit note" : "Add a note"}</Button> : null}
              <Button size="sm" variant="outline" onClick={copy}><CopyIcon />Copy quote</Button>
              {lookupWord(picked.text) ? <Button size="sm" variant="outline" disabled={pending} onClick={define}><BookAIcon />Look up</Button> : null}
              {picked.id ? <Button size="sm" variant="ghost" disabled={pending} onClick={remove}><Trash2Icon />Remove highlight</Button> : null}
              {done ? <span className="text-muted-foreground flex items-center gap-1 text-xs"><CheckIcon className="size-3.5" />{done}</span> : null}
            </div>
            {picked.id && writing === null && marks.find((x) => x.id === picked.id)?.note ? <p className="text-sm"><span className="text-muted-foreground">Your note: </span>{marks.find((x) => x.id === picked.id)!.note}</p> : null}
            {writing !== null ? (
              <div className="space-y-2">
                <Textarea value={writing} onChange={(e) => setWriting(e.target.value)} rows={2} placeholder="Your note on this passage" aria-label="Your note" />
                <div className="flex gap-2">
                  <Button size="sm" disabled={pending} onClick={() => revise({ note: writing })}>Save note</Button>
                  <Button size="sm" variant="ghost" onClick={() => setWriting(null)}>Cancel</Button>
                </div>
              </div>
            ) : null}
            {defined === "none" ? <p className="text-muted-foreground text-sm">Wiktionary has no English entry for that word.</p> : defined ? <Defined found={defined} /> : null}
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


const OLD = /^(archaic|obsolete|dated|historical|poetic|literary)$/;

function Senses({ definitions }: { definitions: Definition[] }) {
  return (
    <>
      {definitions.slice(0, 4).map((d, k) => (
        <div key={k}>
          <p className="text-muted-foreground text-xs italic">{d.pos.toLowerCase()}</p>
          <ol className="list-decimal space-y-1 pl-5">
            {d.senses.slice(0, 5).map((s, j) => (
              <li key={j}>
                {s.old ? <span className="mr-1 rounded bg-amber-100 px-1 text-xs text-amber-900 dark:bg-amber-400/20 dark:text-amber-200">older meaning: {s.labels.filter((l) => OLD.test(l)).join(", ")}</span> : null}
                {s.labels.some((l) => !OLD.test(l)) ? <span className="text-muted-foreground mr-1 text-xs">({s.labels.filter((l) => !OLD.test(l)).join(", ")})</span> : null}
                {s.text}
              </li>
            ))}
          </ol>
        </div>
      ))}
    </>
  );
}

/** A word's senses from Wiktionary, older meanings marked, and the word it is a form of when it is one. */
function Defined({ found }: { found: { word: string; definitions: Definition[]; lemma?: { word: string; definitions: Definition[] } } }) {
  return (
    <div className="space-y-2 rounded-md border p-3 text-sm">
      <p className="font-medium">{found.word}</p>
      <Senses definitions={found.definitions} />
      {found.lemma ? (
        <>
          <p className="pt-1 font-medium">{found.lemma.word}</p>
          <Senses definitions={found.lemma.definitions} />
        </>
      ) : null}
      <p className="text-muted-foreground text-xs">From <a className="underline underline-offset-2" href={`https://en.wiktionary.org/wiki/${encodeURIComponent(found.lemma?.word ?? found.word)}#English`} target="_blank" rel="noreferrer">Wiktionary</a>, CC BY-SA.</p>
    </div>
  );
}
