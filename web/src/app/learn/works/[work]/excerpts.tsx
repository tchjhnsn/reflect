"use client";

import { useState, useTransition } from "react";
import { PlusIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { addExcerptAction } from "../../actions";
import { ExcerptItem, type ExcerptRow } from "../../excerpt-item";

export type { ExcerptRow };

/**
 * The person's excerpts from a book: highlights made while reading and passages they typed with a page or place.
 * Each can be copied as a cited quote, coloured, noted, published or removed; a highlight opens its chapter.
 */
export function Excerpts({ work, rows, reader }: { work: string; rows: ExcerptRow[]; /** Reflect's reader for the book, when it has its text. */ reader?: string }) {
  const [adding, setAdding] = useState(false);
  const [quote, setQuote] = useState("");
  const [place, setPlace] = useState("");
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
          {rows.map((r) => <ExcerptItem key={r.id} row={r} {...(reader ? { reader } : {})} />)}
        </ul>
      ) : null}
      {error ? <p className="text-destructive text-xs">That did not work. Try again.</p> : null}
    </section>
  );
}
