"use client";

import { useRef, useState, useTransition, type KeyboardEvent } from "react";
import { OPENING } from "@/core/journal";
import { drawPrompt, type Prompt } from "@/core/packs";
import { post } from "./actions";

export function Composer({
  today,
  packs,
  used,
}: {
  today: string;
  packs: string[];
  used: string[];
}) {
  const [text, setText] = useState("");
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [turn, setTurn] = useState(0);
  const ask = () => {
    setPrompt(drawPrompt(packs, used, today, turn));
    setTurn(turn + 1);
    ref.current?.focus();
  };
  const [long, setLong] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () =>
    start(async () => {
      const r = await post(text, today, prompt?.id ?? null);
      if ("error" in r) return setError(r.error);
      setText("");
      setLong(false);
      setPrompt(null);
      setError(null);
      ref.current?.focus();
    });

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
  };

  return (
    <form
      className={`composer${long ? " long" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      {prompt ? (
        <div className="composer-prompt" data-testid="prompt">
          <label htmlFor="compose">{prompt.text}</label>
          <button
            type="button"
            onClick={() => setPrompt(null)}
            aria-label="Clear the prompt"
          >
            Clear
          </button>
        </div>
      ) : (
        <label htmlFor="compose" className="sr-only">
          {OPENING}
        </label>
      )}
      <textarea
        id="compose"
        ref={ref}
        value={text}
        placeholder={prompt ? "" : OPENING}
        rows={long ? 12 : 2}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
      />
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        <div className="tools">
          {packs.length > 0 ? (
            <button type="button" onClick={ask}>
              {prompt ? "Another" : "Give me a prompt"}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => setLong(!long)}
            aria-pressed={long}
          >
            {long ? "Shorter" : "Keep writing"}
          </button>
        </div>
        <button
          type="submit"
          className="primary"
          disabled={pending || text.trim() === ""}
        >
          Post
        </button>
      </div>
    </form>
  );
}
