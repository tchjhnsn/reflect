"use client";

import { useState, useTransition } from "react";
import { DOMAINS } from "@reflect/core/book";
import { Button } from "@/components/ui/button";
import { placeAction } from "../actions";

/** Choose the chapter a belief belongs in. */
export function Place({ claimId }: { claimId: string }) {
  const [pending, start] = useTransition();
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {DOMAINS.map((d) => (
        <Button key={d.id} size="sm" variant="outline" className="h-7" disabled={pending} onClick={() => start(async () => {
          const r = await placeAction(claimId, d.id);
          setError("error" in r ? r.error : null);
        })}>{d.title}</Button>
      ))}
      {error ? <span className="text-destructive text-sm" role="alert">{error}</span> : null}
    </div>
  );
}
