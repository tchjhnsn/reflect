"use client";

import { useRef, useState, useTransition, type KeyboardEvent } from "react";
import { OPENING } from "@/core/journal";
import { post } from "./actions";

export function Composer({ today }: { today: string }) {
  const [text, setText] = useState("");
  const [long, setLong] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () =>
    start(async () => {
      const r = await post(text, today, null);
      if ("error" in r) return setError(r.error);
      setText("");
      setLong(false);
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
      <label htmlFor="compose" className="sr-only">
        {OPENING}
      </label>
      <textarea
        id="compose"
        ref={ref}
        value={text}
        placeholder={OPENING}
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
        <button type="button" className="quiet" onClick={() => setLong(!long)} aria-pressed={long}>
          {long ? "Shorter" : "Keep writing"}
        </button>
        <button type="submit" className="primary" disabled={pending || text.trim() === ""}>
          Post
        </button>
      </div>
    </form>
  );
}
