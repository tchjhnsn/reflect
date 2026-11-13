"use client";

import { useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { UploadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { importKindleAction } from "../actions";

/** Bring Kindle highlights in from My Clippings.txt (on a Kindle plugged in by cable, in its documents folder). */
export function KindleImport() {
  const input = useRef<HTMLInputElement>(null);
  const [note, setNote] = useState<string | null>(null);
  const [pending, run] = useTransition();
  const router = useRouter();
  const take = (file: File | undefined) => file && run(async () => {
    const r = await importKindleAction(await file.text());
    if ("error" in r) setNote(r.error);
    else {
      setNote([
        `Added ${r.added} ${r.added === 1 ? "highlight" : "highlights"} from ${r.books} ${r.books === 1 ? "book" : "books"}.`,
        r.already ? `${r.already} were already here.` : "",
        r.notFound.length ? `Not found in the open library: ${r.notFound.slice(0, 5).join("; ")}${r.notFound.length > 5 ? `, and ${r.notFound.length - 5} more` : ""}.` : "",
        r.overLimit ? `${r.overLimit} more were left out; import again to bring them in.` : "",
      ].filter(Boolean).join(" "));
      router.refresh();
    }
    if (input.current) input.current.value = "";
  });
  return (
    <>
      <input ref={input} type="file" accept=".txt,text/plain" hidden onChange={(e) => take(e.target.files?.[0])} />
      <Button variant="outline" size="sm" disabled={pending} onClick={() => input.current?.click()}><UploadIcon />{pending ? "Importing..." : "Import Kindle highlights"}</Button>
      {note ? <p className="text-muted-foreground w-full text-xs" role="status">{note}</p> : null}
    </>
  );
}
