"use client";

import { useState, useTransition } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { addToEntry, writeEntry, type Result } from "./actions";

const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

/** A box to write in: a new entry, or more on an open one when given its id. */
export function Write({ entryId, placeholder, onDone }: { entryId?: string; placeholder: string; onDone?: () => void }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const keep = () =>
    start(async () => {
      const r: Result = entryId ? await addToEntry(entryId, text) : await writeEntry(text, today());
      if ("error" in r) setError(r.error);
      else {
        setText("");
        setError(null);
        onDone?.();
      }
    });
  return (
    <div className="space-y-2">
      <Textarea value={text} onChange={(e) => setText(e.target.value)} placeholder={placeholder} rows={entryId ? 2 : 4} aria-label={placeholder} />
      <div className="flex items-center justify-end gap-3">
        {error ? <p className="text-destructive mr-auto text-sm" role="alert">{error}</p> : null}
        <Button size="sm" onClick={keep} disabled={pending || !text.trim()}>{pending ? "Keeping..." : "Keep"}</Button>
      </div>
    </div>
  );
}
