"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { Domain } from "@reflect/core/book";
import { Button } from "@/components/ui/button";
import { Mark, type IdealOption } from "./mark";

/** Mark a belief already kept an ideal or a practical belief. */
export function MarkLater({ claimId, domain, ideals }: { claimId: string; domain: Domain | null; ideals: IdealOption[] }) {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  if (!open) return <Button size="sm" variant="outline" className="h-7" onClick={() => setOpen(true)}>Ideal or practical belief?</Button>;
  return (
    <div className="rounded-lg border p-3">
      <Mark target={{ claimId }} domain={domain} ideals={ideals.filter((i) => i.id !== claimId)} onCancel={() => setOpen(false)} onDone={() => { setOpen(false); router.refresh(); }} />
    </div>
  );
}
