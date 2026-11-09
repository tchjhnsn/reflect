"use client";

import { useState, type FormEvent } from "react";
import { CHOICES, parseHold } from "@/core/onboarding";
import { HOLDS } from "@/core/structure";
import { CHOICE_LABELS, COPY, HOLD_LABELS } from "./copy";

type Props = {
  id: string;
  initial: string;
  error: string | null;
  today: string;
  canGoBack: boolean;
  onAnswer: (text: string) => void;
  onSkip: () => void;
  onBack: () => void;
};

const ALONE = ["not_for_me", "unsure"];

export function Question({ id, initial, error, today, canGoBack, onAnswer, onSkip, onBack }: Props) {
  const copy = COPY[id] ?? { prompt: id, placeholder: "" };
  const kind = copy.kind ?? (copy.date ? "date" : "text");
  const start = kind === "hold" ? parseHold(initial) : { ids: [], words: "" };
  const [text, setText] = useState(kind === "hold" ? start.words : initial);
  const [picks, setPicks] = useState<string[]>(start.ids);
  const fieldId = `answer-${id}`;
  const errorId = `${fieldId}-error`;
  const promptId = `${fieldId}-prompt`;

  const value = () => (kind === "hold" ? (picks.length ? picks.join(",") + (text.trim() ? `; ${text}` : "") : "") : text);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onAnswer(value());
  };
  const toggle = (h: string) =>
    setPicks((p) => (p.includes(h) ? p.filter((x) => x !== h) : ALONE.includes(h) ? [h] : [...p.filter((x) => !ALONE.includes(x)), h]));

  const described = [copy.note ? `${fieldId}-note` : null, error ? errorId : null].filter(Boolean).join(" ") || undefined;

  return (
    <form className="screen" onSubmit={submit} noValidate>
      {kind === "choice" || kind === "hold" ? (
        <h1 id={promptId} className="prompt">
          {copy.prompt}
        </h1>
      ) : (
        <label htmlFor={fieldId} className="prompt">
          {copy.prompt}
        </label>
      )}
      {kind === "choice" ? (
        <div className="choices" role="radiogroup" aria-labelledby={promptId} aria-describedby={described}>
          {(CHOICES[id] ?? []).map((c, i) => (
            <button key={c} type="button" role="radio" aria-checked={text === c} className="chip" onClick={() => setText(c)} autoFocus={i === 0}>
              {CHOICE_LABELS[c] ?? c}
            </button>
          ))}
        </div>
      ) : kind === "hold" ? (
        <>
          <div className="choices" role="group" aria-labelledby={promptId} aria-describedby={described}>
            {HOLDS.map((h, i) => (
              <button key={h.id} type="button" aria-pressed={picks.includes(h.id)} className="chip" onClick={() => toggle(h.id)} autoFocus={i === 0}>
                {HOLD_LABELS[h.id] ?? h.id}
              </button>
            ))}
          </div>
          <label htmlFor={fieldId} className="sr-only">
            In your own words
          </label>
          <input id={fieldId} className="field" value={text} placeholder={copy.placeholder} onChange={(e) => setText(e.target.value)} autoComplete="off" />
        </>
      ) : (
        <input
          id={fieldId}
          className="field"
          type={kind === "date" ? "date" : "text"}
          inputMode={kind === "number" ? "numeric" : undefined}
          max={kind === "date" ? today : undefined}
          value={text}
          placeholder={copy.placeholder}
          onChange={(e) => setText(e.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={described}
          autoComplete="off"
          autoFocus
        />
      )}
      {copy.note ? (
        <p id={`${fieldId}-note`} className="note">
          {copy.note}
        </p>
      ) : null}
      {error ? (
        <p id={errorId} className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        {canGoBack ? (
          <button type="button" className="quiet" onClick={onBack} aria-label="Back to the previous question">
            Back
          </button>
        ) : (
          <span />
        )}
        <div className="right">
          <button type="button" className="quiet" onClick={onSkip}>
            Skip
          </button>
          <button type="submit" className="primary">
            Next
          </button>
        </div>
      </div>
    </form>
  );
}
