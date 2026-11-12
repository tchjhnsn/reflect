"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import type { ItemSource } from "@reflect/core/items";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { stateItem } from "../actions";

export function Observe({ from }: { from: ItemSource | null }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const router = useRouter();
  const keep = () =>
    start(async () => {
      const r = await stateItem(text, from);
      if ("error" in r) setError(r.error);
      else router.push(`/journal/items/${r.id}`);
    });
  return (
    <div className="space-y-2">
      <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} placeholder="I notice that..." aria-label="What you observe" />
      <div className="flex items-center justify-end gap-3">
        {error ? <p className="text-destructive mr-auto text-sm" role="alert">{error}</p> : null}
        <Button size="sm" onClick={keep} disabled={pending || !text.trim()}>{pending ? "Keeping..." : "State it"}</Button>
      </div>
    </div>
  );
}
