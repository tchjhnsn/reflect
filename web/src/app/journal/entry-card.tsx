"use client";

import { useState, useTransition } from "react";
import { TELL_ME_MORE, type Entry } from "@reflect/core/journal";
import { promptById } from "@reflect/core/packs";
import { done, more } from "./actions";
import { Reflect, Reflections } from "./reflect";
import { elementName } from "@reflect/core/state";

function time(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

export function EntryCard({ entry, label }: { entry: Entry; label?: string }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [reflecting, setReflecting] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const closed = entry.closedAt !== null;
  const prompt = promptById(entry.promptId);
  const finish = () => start(async () => void (await done(entry.id)));

  const add = () =>
    start(async () => {
      const r = await more(entry.id, text);
      if ("error" in r) return setError(r.error);
      setText("");
      setError(null);
    });

  return (
    <article className="entry" data-testid="entry">
      {prompt ? <p className="entry-prompt">{prompt.text}</p> : null}
      {entry.parts.map((p, i) => (
        <p key={p.seq} className={`entry-text${i > 0 ? " more-part" : ""}`}>
          {p.text}
        </p>
      ))}
      {entry.labels && entry.labels.length > 0 ? (
        <p className="entry-labels" data-testid="labels">
          {entry.labels.map(elementName).join(" · ")}
        </p>
      ) : null}
      <Reflections items={(entry.reflections ?? []).filter((r) => r.id !== active)} />
      {reflecting ? (
        <Reflect
          entryId={entry.id}
          onStart={setActive}
          onClose={() => {
            setReflecting(false);
            setActive(null);
          }}
        />
      ) : null}
      {open && !closed ? (
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
            <button type="button" className="quiet" onClick={finish} disabled={pending}>
              I&rsquo;m done
            </button>
            <button type="submit" className="primary" disabled={pending || text.trim() === ""}>
              Add
            </button>
          </div>
        </form>
      ) : null}
      {label ? null : (
        <div className="entry-meta">
          <span>{time(entry.createdAt)}</span>
          {closed ? (
            <>
              <span>Done</span>
              {reflecting ? null : (
                <button type="button" onClick={() => setReflecting(true)}>
                  Reflect on it
                </button>
              )}
            </>
          ) : open ? null : (
            <>
              <button type="button" onClick={() => setOpen(true)}>
                {TELL_ME_MORE}
              </button>
              <button type="button" onClick={finish} disabled={pending}>
                I&rsquo;m done
              </button>
            </>
          )}
        </div>
      )}
    </article>
  );
}
