"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { CheckIcon, CopyIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { citeQuote } from "@reflect/core/excerpts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { addExcerptAction, removeExcerptAction } from "../../actions";

/** One excerpt as the book page shows it: its words, where it is from, and, for a highlight, the chapter to read it in. */
export type ExcerptRow = { id: string; quote: string; where: string | null; chapter: number | null; note: string | null };

/**
 * The person's excerpts from a book: highlights made while reading and passages they typed with a page or place.
 * Each can be copied as a quote cited to the author and book, or removed; a highlight opens its chapter.
 */
export function Excerpts({ work, title, author, rows, reader }: { work: string; title: string; author: string; rows: ExcerptRow[]; /** Reflect's reader for the book, when it has its text. */ reader?: string }) {
  const [adding, setAdding] = useState(false);
  const [quote, setQuote] = useState("");
  const [place, setPlace] = useState("");
  const [copied, setCopied] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const [pending, run] = useTransition();

  const add = () => run(async () => {
    try {
      await addExcerptAction({ work, quote, place });
      setQuote("");
      setPlace("");
      setAdding(false);
      setError(false);
    } catch {
      setError(true);
    }
  });
  const remove = (id: string) => run(async () => {
    try {
      await removeExcerptAction(id, work);
    } catch {
      setError(true);
    }
  });
  const copy = async (r: ExcerptRow) => {
    try {
      await navigator.clipboard.writeText(citeQuote(r.quote, { author, title, ...(r.where ? { place: r.where } : {}) }));
      setCopied(r.id);
    } catch {
      setError(true);
    }
  };

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">Your excerpts{rows.length ? ` · ${rows.length}` : ""}</h3>
        {adding ? null : <Button variant="ghost" size="sm" onClick={() => setAdding(true)}><PlusIcon />Add an excerpt</Button>}
      </div>
      {!rows.length && !adding ? <p className="text-muted-foreground text-sm">{reader ? "Highlight a passage while you read and it is kept here. You can also add one by hand." : "Add a passage from your copy, with its page, and it is kept here."}</p> : null}
      {adding ? (
        <div className="space-y-2 rounded-md border p-3">
          <Textarea value={quote} onChange={(e) => setQuote(e.target.value)} placeholder="The passage, word for word" rows={3} aria-label="The passage" />
          <div className="flex flex-wrap gap-2">
            <Input value={place} onChange={(e) => setPlace(e.target.value)} placeholder="Page or place, such as p. 12" className="h-8 max-w-56" aria-label="Page or place" />
            <Button size="sm" disabled={pending || !quote.trim()} onClick={add}>Keep it</Button>
            <Button size="sm" variant="ghost" onClick={() => setAdding(false)}>Cancel</Button>
          </div>
        </div>
      ) : null}
      {rows.length ? (
        <ul className="space-y-3">
          {rows.map((r) => (
            <li key={r.id} className="border-l-2 border-amber-300 pl-3">
              <blockquote className="font-serif text-[15px] leading-relaxed">{r.quote}</blockquote>
              <div className="text-muted-foreground mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                {r.where ? reader && r.chapter ? <Link className="underline-offset-4 hover:underline" href={`${reader}?c=${r.chapter}`}>{r.where}</Link> : <span>{r.where}</span> : null}
                <button type="button" className="hover:text-foreground inline-flex items-center gap-1" onClick={() => copy(r)}>{copied === r.id ? <CheckIcon className="size-3" /> : <CopyIcon className="size-3" />}{copied === r.id ? "Copied" : "Copy quote"}</button>
                <button type="button" className="hover:text-foreground inline-flex items-center gap-1" disabled={pending} onClick={() => remove(r.id)}><Trash2Icon className="size-3" />Remove</button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
      {error ? <p className="text-destructive text-xs">That did not work. Try again.</p> : null}
    </section>
  );
}
