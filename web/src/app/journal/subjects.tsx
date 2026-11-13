"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import type { Domain } from "@reflect/core/book";
import { Button } from "@/components/ui/button";
import { placeSubjectAction } from "./actions";

const select = "border-input bg-background h-8 rounded-md border px-2 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50";

/** Bring a Learn subject into this domain. */
export function AddSubject({ domain, subjects }: { domain: Domain; subjects: { id: string; name: string }[] }) {
  const [pick, setPick] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, run] = useTransition();
  const router = useRouter();
  if (!subjects.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <select aria-label="Subject from Learn" className={select} value={pick} onChange={(e) => setPick(e.target.value)}>
        <option value="">A subject from Learn</option>
        {subjects.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
      </select>
      <Button size="sm" variant="outline" className="h-8" disabled={!pick || pending} onClick={() => run(async () => {
        const r = await placeSubjectAction(pick, domain);
        if ("error" in r) setError(r.error);
        else { setPick(""); setError(null); router.refresh(); }
      })}>Add</Button>
      {error ? <span className="text-destructive text-sm" role="alert">{error}</span> : null}
    </div>
  );
}

/** Take a subject out of this domain; it stays in Learn. */
export function RemoveSubject({ id, name }: { id: string; name: string }) {
  const [pending, run] = useTransition();
  const router = useRouter();
  return (
    <Button size="sm" variant="ghost" className="text-muted-foreground h-7" aria-label={`Take ${name} out of this domain`} disabled={pending}
      onClick={() => run(async () => { await placeSubjectAction(id, null); router.refresh(); })}>Take out</Button>
  );
}
