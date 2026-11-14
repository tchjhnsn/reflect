"use client";

import { useState, useTransition } from "react";
import { VideoIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { examCallAction } from "./actions";

/** The examination's call inside Reflect: joined with a button, so camera and microphone are asked for only then. */
export function ExamCall({ id }: { id: string }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [pending, run] = useTransition();
  if (url) {
    return (
      <div className="space-y-2">
        <iframe src={url} title="The examination's call" allow="camera; microphone; fullscreen; display-capture; autoplay; speaker-selection" className="aspect-video w-full rounded-lg border bg-black" />
        <Button type="button" size="sm" variant="outline" onClick={() => setUrl(null)}>Leave the call</Button>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button type="button" disabled={pending} onClick={() => run(async () => { const r = await examCallAction(id).catch(() => null); if (r) setUrl(r.url); else setFailed(true); })}><VideoIcon />Join the call</Button>
      <span className="text-muted-foreground text-sm">{failed ? "The call did not open. Try again in a moment." : "Nothing is recorded."}</span>
    </div>
  );
}
