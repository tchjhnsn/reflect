"use client";

import { useState, useTransition } from "react";
import { STANCES } from "@reflect/core/claims";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { sendToLearnAction } from "./actions";

/** Send a passage to Learn as a claim: the subject it belongs under, then your stance on it. */
export function SendToLearn({ workId, passage, subjects, onDone, onCancel }: {
  workId: string;
  passage: string;
  subjects: string[];
  onDone: (subject: string) => void;
  onCancel: () => void;
}) {
  const [subject, setSubject] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, run] = useTransition();
  const send = (stance: string) => run(async () => {
    const r = await sendToLearnAction(workId, passage, subject, stance);
    if ("error" in r) setError(r.error);
    else onDone(subject.trim());
  });
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <Input autoFocus list="learn-subjects" value={subject} onChange={(e) => setSubject(e.target.value)} placeholder="Subject in Learn" aria-label="Subject" className="h-8 w-56" />
      <datalist id="learn-subjects">{subjects.map((s) => <option key={s} value={s} />)}</datalist>
      {STANCES.map((s) => <Button key={s.id} type="button" size="sm" variant="outline" className="h-8" disabled={pending || !subject.trim()} onClick={() => send(s.id)}>{s.text}</Button>)}
      <Button type="button" size="sm" variant="ghost" className="h-8" onClick={onCancel}>Cancel</Button>
      {error ? <span className="text-destructive" role="alert">{error}</span> : null}
    </div>
  );
}
