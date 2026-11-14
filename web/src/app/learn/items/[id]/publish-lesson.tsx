"use client";

import { useState, useTransition } from "react";
import { Button } from "@/components/ui/button";
import { publishLessonAction } from "../../actions";

/** Publish the latest lesson at its own public page, after the person confirms what will be shown. */
export function PublishLesson({ id }: { id: string }) {
  const [step, setStep] = useState<"idle" | "confirm">("idle");
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, run] = useTransition();
  if (url) return <p className="text-sm" role="status">Published at <a href={url} className="underline underline-offset-4">{url}</a>. Take it down on this page.</p>;
  if (step === "idle") return <Button type="button" size="sm" variant="ghost" onClick={() => setStep("confirm")}>Publish the latest lesson</Button>;
  return (
    <div className="space-y-2 rounded-md border p-3 text-sm">
      <p>Anyone with the link will see the lesson with this item&apos;s title and maker. Nothing else is shown.</p>
      <div className="flex gap-2">
        <Button type="button" size="sm" disabled={pending} onClick={() => run(async () => { try { setUrl(`/q/${await publishLessonAction(id)}`); } catch (e) { setError((e as Error).message); } })}>Publish</Button>
        <Button type="button" size="sm" variant="ghost" onClick={() => setStep("idle")}>Cancel</Button>
      </div>
      {error ? <p className="text-destructive text-xs" role="alert">{error}</p> : null}
    </div>
  );
}
