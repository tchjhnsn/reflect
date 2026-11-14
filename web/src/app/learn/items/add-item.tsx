"use client";

import { useState, useTransition } from "react";
import { SearchIcon } from "lucide-react";
import { MEDIA, WAYS, mediumOf, type Medium } from "@reflect/core/ways";
import type { Found } from "@reflect/core/lookup";
import { SEARCHABLE } from "@reflect/core/lookup";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { addItemAction, lookLinkAction, searchItemsAction } from "../actions";
import { LocalDay } from "./local-day";

const SELECT = "border-input bg-background h-9 w-full rounded-md border px-3 text-sm";
const looksLikeLink = (s: string) => /^https?:\/\//i.test(s.trim());

/**
 * Add a lecture, debate, film, music, game and the rest: paste its link or search the catalog for its medium,
 * pick what was found, and the title, maker, link and picture are filled in. Everything stays editable.
 */
export function AddItem({ books, medium: start = "lecture" }: { books: { work: string; title: string }[]; medium?: Medium }) {
  const [medium, setMedium] = useState<Medium>(start);
  const [find, setFind] = useState("");
  const [found, setFound] = useState<Found[]>([]);
  const [note, setNote] = useState<string | null>(null);
  const [f, setF] = useState({ title: "", by: "", url: "", image: "" });
  const [pending, run] = useTransition();
  const pick = (x: Found) => {
    setMedium(x.medium);
    setF({ title: x.title, by: x.by ?? "", url: x.url ?? "", image: x.image ?? "" });
    setFound([]);
    setNote(`Filled in from ${x.source}. Change anything before adding.`);
  };
  const go = () => run(async () => {
    setNote(null);
    setFound([]);
    if (looksLikeLink(find)) {
      const r = await lookLinkAction(find.trim());
      if (r.found) pick(r.found);
      else {
        setF((v) => ({ ...v, url: find.trim() }));
        setNote(r.error ?? "Nothing found for that link.");
      }
      return;
    }
    const r = await searchItemsAction(medium, find);
    setFound(r.found);
    setNote(r.error ?? (r.found.length ? null : "Nothing found. Fill it in by hand."));
  });
  const m = mediumOf(medium);
  return (
    <Card className="py-5" id="add">
      <CardHeader className="px-5"><CardTitle className="text-sm">Add a lecture, debate, film, music or game</CardTitle></CardHeader>
      <CardContent className="space-y-4 px-5">
        <div className="grid gap-2 sm:grid-cols-[10rem_1fr_auto]">
          <select aria-label="Medium" className={SELECT} value={medium} onChange={(e) => setMedium(e.target.value as Medium)}>
            {WAYS.map((w) => <optgroup key={w.id} label={w.label}>{MEDIA.filter((x) => x.way === w.id).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}</optgroup>)}
          </select>
          <Input aria-label="Paste a link or search" placeholder={SEARCHABLE.includes(medium) ? "Paste a link, or search by title" : "Paste a link (YouTube, Vimeo, Spotify, Apple Podcasts)"} value={find} onChange={(e) => setFind(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); go(); } }} />
          <Button type="button" variant="outline" disabled={pending || !find.trim()} onClick={go}><SearchIcon />{pending ? "Finding..." : "Find"}</Button>
        </div>
        {note ? <p className="text-muted-foreground text-xs" role="status">{note}</p> : null}
        {found.length ? (
          <ul className="grid gap-2 sm:grid-cols-2" aria-label="Found">
            {found.map((x, i) => (
              <li key={`${x.url ?? x.title}-${i}`}>
                <button type="button" onClick={() => pick(x)} className="hover:bg-accent flex w-full items-center gap-3 rounded-lg border p-2 text-left transition">
                  {x.image ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={x.image} alt="" loading="lazy" className="h-14 w-14 shrink-0 rounded object-cover" />
                  ) : <span className="bg-muted h-14 w-14 shrink-0 rounded" />}
                  <span className="min-w-0">
                    <span className="line-clamp-2 block text-sm font-medium">{x.title}</span>
                    <span className="text-muted-foreground line-clamp-1 block text-xs">{[x.by, x.year, x.source].filter(Boolean).join(" · ")}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        <form action={addItemAction} className="grid gap-4 sm:grid-cols-2" onSubmit={() => setTimeout(() => { setF({ title: "", by: "", url: "", image: "" }); setFind(""); setNote(null); }, 0)}>
          <input type="hidden" name="medium" value={medium} />
          <input type="hidden" name="image" value={f.image} />
          <LocalDay />
          <div className="flex gap-3 sm:col-span-2">
            {f.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={f.image} alt="" className="h-20 w-20 shrink-0 rounded object-cover" />
            ) : null}
            <label className="block flex-1 space-y-1.5 text-sm font-medium">Title<Input name="title" required maxLength={300} value={f.title} onChange={(e) => setF({ ...f, title: e.target.value })} /></label>
          </div>
          <label className="block space-y-1.5 text-sm font-medium">By (optional)<Input name="by" maxLength={200} value={f.by} onChange={(e) => setF({ ...f, by: e.target.value })} /></label>
          <label className="block space-y-1.5 text-sm font-medium">Link (optional)<Input name="url" type="url" maxLength={1000} value={f.url} onChange={(e) => setF({ ...f, url: e.target.value })} /></label>
          <label className="block space-y-1.5 text-sm font-medium">Status
            <select name="status" className={SELECT} defaultValue="done">
              <option value="done">{m.done}</option>
              <option value="doing">{m.doing}</option>
              <option value="want">Want to</option>
            </select>
          </label>
          <label className="block space-y-1.5 text-sm font-medium">Way of learning
            <select name="way" className={SELECT} defaultValue="">
              <option value="">The medium&apos;s way ({WAYS.find((w) => w.id === m.way)!.label})</option>
              {WAYS.map((w) => <option key={w.id} value={w.id}>{w.label}</option>)}
            </select>
          </label>
          {medium === "course" ? <label className="block space-y-1.5 text-sm font-medium">How many parts (optional)<Input name="parts" type="number" min={1} max={10000} /></label> : null}
          {medium === "debate" || medium === "discussion" ? (
            <label className="flex items-center gap-2 self-end text-sm"><input type="checkbox" name="took_part" className="size-4" />I took part</label>
          ) : null}
          {books.length && (m.way === "imitation" || medium === "theatre") ? (
            <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">Adapts or stages a book on your shelves (optional)
              <select name="work" className={SELECT} defaultValue="">
                <option value="">None</option>
                {books.map((b) => <option key={b.work} value={b.work}>{b.title}</option>)}
              </select>
            </label>
          ) : null}
          <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">What it taught you (optional)<Textarea name="lesson" rows={3} maxLength={4000} /></label>
          <p className="text-muted-foreground text-xs sm:col-span-2">A game, music or art is kept under Rest and leisure until you say what it taught you.</p>
          <div className="sm:col-span-2"><Button size="sm">Add</Button></div>
        </form>
      </CardContent>
    </Card>
  );
}
