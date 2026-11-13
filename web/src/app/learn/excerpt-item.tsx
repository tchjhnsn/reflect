"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { CheckIcon, CopyIcon, GlobeIcon, MessageSquareIcon, Trash2Icon } from "lucide-react";
import { citeQuote, HIGHLIGHT_COLORS, type HighlightColor } from "@reflect/core/excerpts";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { publishQuoteAction, removeExcerptAction, reviseExcerptAction, unpublishQuoteAction } from "./actions";
import { COLOR } from "./colors";

/** One excerpt as a list shows it: its words, colour, note, where it is from, and its page when published. */
export type ExcerptRow = { id: string; work: string; title: string; author: string; quote: string; where: string | null; chapter: number | null; note: string | null; color: HighlightColor; published: string | null };

/**
 * One excerpt with what can be done to it: copy it as a cited quote, change its colour, write or change its note,
 * publish it at its own page (only after the person confirms) or take it down, and remove it.
 */
export function ExcerptItem({ row, reader }: { row: ExcerptRow; /** Reflect's reader for the book, when it has its text. */ reader?: string }) {
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [note, setNote] = useState(row.note ?? "");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(false);
  const [pending, run] = useTransition();
  const cite = citeQuote(row.quote, { author: row.author, title: row.title, ...(row.where ? { place: row.where } : {}) });
  const act = (f: () => Promise<unknown>) => run(async () => {
    try {
      await f();
      setError(false);
    } catch {
      setError(true);
    }
  });
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cite);
      setCopied(true);
    } catch {
      setError(true);
    }
  };
  return (
    <li className={cn("border-l-2 pl-3", COLOR[row.color].bar)}>
      <blockquote className="font-serif text-[15px] leading-relaxed">{row.quote}</blockquote>
      {row.note && !editing ? <p className="mt-1 text-sm">{row.note}</p> : null}
      {editing ? (
        <div className="mt-2 space-y-2">
          <Textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} placeholder="Your note" aria-label="Your note" />
          <div className="flex gap-2">
            <Button size="sm" disabled={pending} onClick={() => act(async () => { await reviseExcerptAction(row.id, row.work, { note }); setEditing(false); })}>Save note</Button>
            <Button size="sm" variant="ghost" onClick={() => { setEditing(false); setNote(row.note ?? ""); }}>Cancel</Button>
          </div>
        </div>
      ) : null}
      <div className="text-muted-foreground mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        {row.where ? reader && row.chapter ? <Link className="underline-offset-4 hover:underline" href={`${reader}?c=${row.chapter}`}>{row.where}</Link> : <span>{row.where}</span> : null}
        <span className="inline-flex items-center gap-1" role="group" aria-label="Colour">
          {HIGHLIGHT_COLORS.map((c) => (
            <button key={c} type="button" aria-label={COLOR[c].name} aria-pressed={c === row.color} disabled={pending || c === row.color} onClick={() => act(() => reviseExcerptAction(row.id, row.work, { color: c }))} className={cn("size-3 rounded-full", COLOR[c].dot, c === row.color ? "ring-foreground/60 ring-2 ring-offset-1" : "opacity-60 hover:opacity-100")} />
          ))}
        </span>
        <button type="button" className="hover:text-foreground inline-flex items-center gap-1" onClick={copy}>{copied ? <CheckIcon className="size-3" /> : <CopyIcon className="size-3" />}{copied ? "Copied" : "Copy quote"}</button>
        {editing ? null : <button type="button" className="hover:text-foreground inline-flex items-center gap-1" onClick={() => setEditing(true)}><MessageSquareIcon className="size-3" />{row.note ? "Edit note" : "Add a note"}</button>}
        {row.published ? (
          <>
            <Link href={`/q/${row.published}`} className="text-foreground inline-flex items-center gap-1 underline-offset-4 hover:underline"><GlobeIcon className="size-3" />Published</Link>
            <button type="button" className="hover:text-foreground" disabled={pending} onClick={() => act(() => unpublishQuoteAction(row.published!))}>Take down</button>
          </>
        ) : (
          <button type="button" className="hover:text-foreground inline-flex items-center gap-1" onClick={() => setAsking(true)}><GlobeIcon className="size-3" />Publish</button>
        )}
        <button type="button" className="hover:text-foreground inline-flex items-center gap-1" disabled={pending} onClick={() => act(() => removeExcerptAction(row.id, row.work))}><Trash2Icon className="size-3" />Remove</button>
      </div>
      {asking ? (
        <div className="mt-2 space-y-2 rounded-md border p-3 text-sm">
          <p>Publish this quote? Anyone with its link will see the quote and its citation, “{row.author ? `${row.author}, ` : ""}{row.title}{row.where ? `, ${row.where}` : ""}”. Your note and your other highlights stay private.</p>
          <div className="flex gap-2">
            <Button size="sm" disabled={pending} onClick={() => act(async () => { await publishQuoteAction({ quote: row.quote, author: row.author, title: row.title, place: row.where, work: row.work }); setAsking(false); })}>Publish</Button>
            <Button size="sm" variant="ghost" onClick={() => setAsking(false)}>Cancel</Button>
          </div>
        </div>
      ) : null}
      {error ? <p className="text-destructive mt-1 text-xs">That did not work. Try again.</p> : null}
    </li>
  );
}
