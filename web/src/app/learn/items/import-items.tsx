"use client";

import { useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { UploadIcon } from "lucide-react";
import { IMPORT_KINDS, notKept, readImport, type Candidate, type ImportKind } from "@reflect/core/item-imports";
import { MEDIA, mediumOf, type Medium } from "@reflect/core/ways";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { importItemsAction } from "../actions";

const SELECT = "border-input bg-background h-9 max-w-full min-w-0 rounded-md border px-3 text-sm";

/**
 * Bring items in from another service's export: the file is read in the browser, what you already keep is left
 * out, and you pick which to keep. Nothing is sent until you press Keep.
 */
export function ImportItems({ kept }: { kept: { medium: string; title: string }[] }) {
  const input = useRef<HTMLInputElement>(null);
  const [kind, setKind] = useState<ImportKind>("letterboxd");
  const [medium, setMedium] = useState<Medium>("lecture");
  const [list, setList] = useState<Candidate[]>([]);
  const [on, setOn] = useState<Set<number>>(new Set());
  const [note, setNote] = useState<string | null>(null);
  const [pending, run] = useTransition();
  const router = useRouter();
  const read = async (file: File | undefined) => {
    if (!file) return;
    try {
      const all = readImport(kind, await file.text(), medium);
      const fresh = notKept(all, kept);
      setList(fresh);
      setOn(new Set(kind === "youtube" || kind === "spotify" ? [] : fresh.map((_, i) => i)));
      setNote(fresh.length ? `${fresh.length} found${all.length > fresh.length ? `, ${all.length - fresh.length} you already keep left out` : ""}. Pick which to keep.` : "Nothing new in that file.");
    } catch (e) {
      setNote((e as Error).message);
      setList([]);
    }
    if (input.current) input.current.value = "";
  };
  const keep = () => run(async () => {
    const n = await importItemsAction(list.filter((_, i) => on.has(i)));
    setNote(`Kept ${n}.`);
    setList([]);
    router.refresh();
  });
  const k = IMPORT_KINDS.find((x) => x.id === kind)!;
  return (
    <Card className="py-5">
      <CardHeader className="px-5"><CardTitle className="text-sm">Import from another service</CardTitle></CardHeader>
      <CardContent className="space-y-3 px-5">
        <div className="flex flex-wrap items-center gap-2">
          <select aria-label="From" className={SELECT} value={kind} onChange={(e) => { setKind(e.target.value as ImportKind); setList([]); setNote(null); }}>
            {IMPORT_KINDS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
          {kind === "lines" ? (
            <select aria-label="Medium" className={SELECT} value={medium} onChange={(e) => setMedium(e.target.value as Medium)}>
              {MEDIA.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          ) : null}
          <input ref={input} type="file" accept={k.accept} hidden onChange={(e) => read(e.target.files?.[0])} />
          <Button type="button" variant="outline" size="sm" onClick={() => input.current?.click()}><UploadIcon />Choose file</Button>
        </div>
        {note ? <p className="text-muted-foreground text-xs" role="status">{note}</p> : null}
        {list.length ? (
          <>
            <div className="flex flex-wrap gap-2">
              <Button type="button" size="sm" variant="ghost" onClick={() => setOn(new Set(list.map((_, i) => i)))}>Select all</Button>
              <Button type="button" size="sm" variant="ghost" onClick={() => setOn(new Set())}>Select none</Button>
              <Button type="button" size="sm" disabled={pending || !on.size} onClick={keep}>{pending ? "Keeping..." : `Keep ${on.size}`}</Button>
            </div>
            <ul className="max-h-80 space-y-1 overflow-y-auto rounded-md border p-2">
              {list.map((c, i) => (
                <li key={i}>
                  <label className="hover:bg-accent flex items-start gap-2 rounded px-2 py-1 text-sm">
                    <input type="checkbox" className="mt-0.5 size-4" checked={on.has(i)} onChange={(e) => { const n = new Set(on); if (e.target.checked) n.add(i); else n.delete(i); setOn(n); }} />
                    <span className="min-w-0"><span className="block">{c.title}</span><span className="text-muted-foreground block text-xs">{[mediumOf(c.medium).label, c.by, c.note, c.on].filter(Boolean).join(" · ")}</span></span>
                  </label>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
