"use client";

import { useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { UploadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { importGoodreadsAction } from "./actions";

/**
 * Bring a Goodreads library in from its export file (Goodreads: My Books, Import and export, Export library).
 * Goodreads no longer gives apps access, so the export is the way in.
 */
export function GoodreadsImport() {
  const input = useRef<HTMLInputElement>(null);
  const [note, setNote] = useState<string | null>(null);
  const [pending, run] = useTransition();
  const router = useRouter();
  const take = (file: File | undefined) => file && run(async () => {
    const r = await importGoodreadsAction(await file.text());
    if ("error" in r) setNote(r.error);
    else {
      setNote([
        `Added ${r.added} ${r.added === 1 ? "book" : "books"} to your shelves.`,
        r.already ? `${r.already} were already there.` : "",
        r.uncataloged ? `${r.uncataloged} not found in the open library, kept unshelved.` : "",
        r.overLimit ? `${r.overLimit} more were left out; import again to bring them in.` : "",
      ].filter(Boolean).join(" "));
      router.refresh();
    }
    if (input.current) input.current.value = "";
  });
  return (
    <div className="flex flex-wrap items-center gap-2">
      <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => take(e.target.files?.[0])} />
      <Button variant="outline" className="h-10" disabled={pending} onClick={() => input.current?.click()}><UploadIcon />{pending ? "Importing from Goodreads..." : "Import from Goodreads"}</Button>
      {note ? <span className="text-muted-foreground text-xs" role="status">{note}</span> : null}
    </div>
  );
}
