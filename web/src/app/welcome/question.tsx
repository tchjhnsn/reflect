"use client";

import { useState, type FormEvent, type KeyboardEvent } from "react";
import { COPY } from "./copy";

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

export function Question({ id, initial, error, today, canGoBack, onAnswer, onSkip, onBack }: Props) {
  const copy = COPY[id] ?? { prompt: id, placeholder: "" };
  const [text, setText] = useState(initial);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onAnswer(text);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onAnswer(text);
  };
  const fieldId = `answer-${id}`;
  const errorId = `${fieldId}-error`;

  return (
    <form className="screen" onSubmit={submit} noValidate>
      <label htmlFor={fieldId} className="prompt">
        {copy.prompt}
      </label>
      {copy.long ? (
        <textarea
          id={fieldId}
          className="field long"
          value={text}
          placeholder={copy.placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          rows={4}
          autoFocus
        />
      ) : (
        <input
          id={fieldId}
          className="field"
          type={copy.date ? "date" : "text"}
          max={copy.date ? today : undefined}
          value={text}
          placeholder={copy.placeholder}
          onChange={(e) => setText(e.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errorId : undefined}
          autoComplete="off"
          autoFocus
        />
      )}
      {copy.note ? <p className="note">{copy.note}</p> : null}
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
