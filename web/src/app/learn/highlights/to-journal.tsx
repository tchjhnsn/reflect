"use client";

import { useTransition } from "react";
import { useRouter } from "next/navigation";
import { NotebookPenIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { highlightsToJournalAction } from "../actions";

/** Put all highlights into the journal as one work, and open it. */
export function ToJournal() {
  const [pending, run] = useTransition();
  const router = useRouter();
  return (
    <Button variant="outline" size="sm" disabled={pending} onClick={() => run(async () => router.push(`/journal/w/${await highlightsToJournalAction()}`))}>
      <NotebookPenIcon />{pending ? "Sending..." : "Send to journal"}
    </Button>
  );
}
