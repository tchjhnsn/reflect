"use client";

import { useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { UploadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { importWorkAction } from "./actions";

/** Bring Markdown or text files in as works, one at a time, saying what did not come in. */
export function ImportButton() {
  const input = useRef<HTMLInputElement>(null);
  const [pending, start] = useTransition();
  const [note, setNote] = useState<string | null>(null);
  const router = useRouter();
  const take = (files: FileList | null) =>
    start(async () => {
      const failed: string[] = [];
      let done = 0;
      for (const f of Array.from(files ?? [])) {
        const r = await importWorkAction(f.name, await f.text());
        if ("error" in r) failed.push(r.error);
        else done += 1;
      }
      setNote([done ? `Imported ${done} ${done === 1 ? "file" : "files"}.` : "", ...failed].filter(Boolean).join(" "));
      if (input.current) input.current.value = "";
      router.refresh();
    });
  return (
    <>
      {note ? <span className="text-muted-foreground max-w-56 truncate text-xs" role="status" title={note}>{note}</span> : null}
      <input ref={input} type="file" accept=".md,.markdown,.txt,.text,text/markdown,text/plain" multiple hidden onChange={(e) => take(e.target.files)} />
      <Button variant="outline" disabled={pending} onClick={() => input.current?.click()}><UploadIcon />{pending ? "Importing..." : "Import"}</Button>
    </>
  );
}
