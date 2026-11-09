"use client";

import { useState, useTransition } from "react";
import { FRAMES, frameById, type FrameId } from "@/core/reflect";
import type { Reflection } from "@/core/journal";
import { beginReflection, reflectAnswer } from "./actions";

/** Pick a frame, then fixed open questions one at a time. Skipping writes nothing. No model is asked. */
export function Reflect({ entryId, onClose, onStart }: { entryId: string; onClose: () => void; onStart: (id: string) => void }) {
  const [frame, setFrame] = useState<FrameId | null>(null);
  const [id, setId] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const f = frame ? frameById(frame) : null;
  const q = f?.questions[step] ?? null;

  const pick = (fid: FrameId) =>
    start(async () => {
      const r = await beginReflection(entryId, fid);
      if ("error" in r) return setError(r.error);
      setError(null);
      setFrame(fid);
      setId(r.id);
      onStart(r.id);
    });

  const next = (words: string) =>
    start(async () => {
      if (!id) return;
      const r = await reflectAnswer(id, step, words);
      if ("error" in r) return setError(r.error);
      setError(null);
      setText("");
      setStep(step + 1);
    });

  if (!f || !id) {
    return (
      <div className="reflect" role="group" aria-labelledby={`reflect-${entryId}`}>
        <p id={`reflect-${entryId}`} className="reflect-title">
          What would you like to look at?
        </p>
        <div className="choices">
          {FRAMES.map((fr) => (
            <button key={fr.id} type="button" className="chip" onClick={() => pick(fr.id)} disabled={pending}>
              {fr.title}
              <span className="chip-note">{fr.about}</span>
            </button>
          ))}
        </div>
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        <div className="actions">
          <button type="button" className="quiet" onClick={onClose}>
            Not now
          </button>
        </div>
      </div>
    );
  }

  if (!q) {
    return (
      <div className="reflect">
        <p className="reflect-title">That&rsquo;s every question for {f.title.toLowerCase()}.</p>
        <div className="actions">
          <button type="button" className="primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <form
      className="reflect"
      onSubmit={(e) => {
        e.preventDefault();
        next(text);
      }}
    >
      <p className="reflect-step">
        {f.title} &middot; {step + 1} of {f.questions.length}
      </p>
      <label htmlFor={`reflect-q-${entryId}`} className="more-prompt">
        {q.text}
      </label>
      <textarea id={`reflect-q-${entryId}`} value={text} rows={3} onChange={(e) => setText(e.target.value)} autoFocus />
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        <button type="button" className="quiet" onClick={onClose} disabled={pending}>
          Stop here
        </button>
        <button type="button" className="quiet" onClick={() => next("")} disabled={pending}>
          Skip
        </button>
        <button type="submit" className="primary" disabled={pending || text.trim() === ""}>
          Next
        </button>
      </div>
    </form>
  );
}

/** A kept reflection: what happened apart from what the person felt they ought. */
export function Reflections({ items }: { items: readonly Reflection[] }) {
  const shown = items.filter((r) => r.answers.length > 0);
  if (shown.length === 0) return null;
  return (
    <div className="reflections">
      {shown.map((r) => {
        const is = r.answers.filter((a) => a.stance === "descriptive");
        const ought = r.answers.filter((a) => a.stance === "normative");
        return (
          <section key={r.id} className="reflection" data-testid="reflection" aria-label={`Reflection: ${frameById(r.frame)?.title ?? r.frame}`}>
            <p className="reflect-step">Reflected on {frameById(r.frame)?.title.toLowerCase() ?? r.frame}</p>
            {is.length > 0 ? (
              <dl data-testid="reflection-is">
                {is.map((a) => (
                  <div key={a.seq}>
                    <dt>{a.question}</dt>
                    <dd>{a.text}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
            {ought.length > 0 ? (
              <>
                <p className="reflect-step">What you felt you ought</p>
                <dl data-testid="reflection-ought">
                  {ought.map((a) => (
                    <div key={a.seq}>
                      <dt>{a.question}</dt>
                      <dd>{a.text}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
