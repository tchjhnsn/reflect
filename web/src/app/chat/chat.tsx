"use client";

import { useMemo, useRef, useState } from "react";
import type { Spec } from "@json-render/core";
import { createStateStore } from "@json-render/core";
import { JSONUIProvider, Renderer, type SetState } from "@json-render/react";
import { askAction, type Turn } from "./actions";
import { handlers, registry } from "./registry";

type Shown = Turn & { spec?: Spec | null };

/** One answer's cards, each answer with its own state so a tap marks only its card. */
function Cards({ spec }: { spec: Spec }) {
  const store = useMemo(() => createStateStore(spec.state ?? {}), [spec]);
  const acts = useMemo(() => {
    const setState: SetState = (f) => {
      const prev = store.getSnapshot();
      const next = f(prev);
      store.update(Object.fromEntries(Object.keys(next).filter((k) => next[k] !== prev[k]).map((k) => [`/${k}`, next[k]])));
    };
    return handlers(() => setState, () => store.getSnapshot());
  }, [store]);
  return (
    <JSONUIProvider registry={registry} store={store} handlers={acts}>
      <Renderer spec={spec} registry={registry} />
    </JSONUIProvider>
  );
}

export function Chat() {
  const [turns, setTurns] = useState<Shown[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const said = text.trim();
    if (!said || busy) return;
    const next: Shown[] = [...turns, { role: "user", content: said }];
    setTurns(next);
    setText("");
    setBusy(true);
    try {
      const a = await askAction(next.map(({ role, content }) => ({ role, content })));
      setTurns([...next, { role: "assistant", content: a.say, spec: a.spec }]);
    } catch {
      setTurns([...next, { role: "assistant", content: "Something went wrong on my side. Try again." }]);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => end.current?.scrollIntoView({ behavior: "smooth" }));
    }
  }

  return (
    <div className="chat">
      <div className="chat-log" aria-live="polite">
        {turns.length ? null : <p className="muted chat-empty">Ask about what you are learning: your shelves, a book to find, a claim you hold. I show what is in your apps, and nothing is kept unless you accept it.</p>}
        {turns.map((t, i) => (
          <div key={i} className={`chat-msg chat-${t.role}`}>
            <p>{t.content}</p>
            {t.spec ? <Cards spec={t.spec} /> : null}
          </div>
        ))}
        {busy ? <p className="muted">Looking in your apps…</p> : null}
        <div ref={end} />
      </div>
      <form className="chat-form" onSubmit={send}>
        <label htmlFor="chat-input" className="sr-only">Message</label>
        <input id="chat-input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask Reflect" autoComplete="off" maxLength={4000} />
        <button type="submit" className="primary" disabled={busy || !text.trim()}>Send</button>
      </form>
    </div>
  );
}
