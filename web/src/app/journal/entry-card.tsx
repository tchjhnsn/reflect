"use client";

import { useState, useTransition } from "react";
import { TELL_ME_MORE, type Entry } from "@/core/journal";
import { promptById } from "@/core/packs";
import { done, more } from "./actions";

function time(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

export function EntryCard({ entry, label }: { entry: Entry; label?: string }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const closed = entry.closedAt !== null;

  const add = () =>
    start(async () => {
      const r = await more(entry.id, text);
      if ("error" in r) return setError(r.error);
      setText("");
      setError(null);
    });

  return (
    <article className="entry" data-testid="entry">
      <header className="entry-meta">
        <span>{label ?? time(entry.createdAt)}</span>
        {closed ? <span className="badge">Done</span> : null}
      </header>
      {promptById(entry.promptId) ? <p className="entry-prompt">{promptById(entry.promptId)!.text}</p> : null}
      {entry.parts.map((p) => (
        <p key={p.seq} className="entry-text">
          {p.text}
        </p>
      ))}
      {closed || label ? null : open ? (
        <form
          className="more"
          onSubmit={(e) => {
            e.preventDefault();
            add();
          }}
        >
          <label htmlFor={`more-${entry.id}`} className="more-prompt">
            {TELL_ME_MORE}
          </label>
          <textarea id={`more-${entry.id}`} value={text} rows={3} onChange={(e) => setText(e.target.value)} autoFocus />
          {error ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}
          <div className="actions">
            <button type="button" className="quiet" onClick={() => start(async () => void (await done(entry.id)))} disabled={pending}>
              I&rsquo;m done
            </button>
            <button type="submit" className="primary" disabled={pending || text.trim() === ""}>
              Add
            </button>
          </div>
        </form>
      ) : (
        <div className="entry-actions">
          <button type="button" className="quiet" onClick={() => setOpen(true)}>
            {TELL_ME_MORE}
          </button>
          <button type="button" className="quiet" onClick={() => start(async () => void (await done(entry.id)))} disabled={pending}>
            I&rsquo;m done
          </button>
        </div>
      )}
    </article>
  );
}
