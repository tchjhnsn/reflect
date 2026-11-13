"use client";

// The chat, after json-render's chat example (vercel-labs/json-render, examples/chat): AI SDK useChat over
// /api/chat, parts kept in order (text, tool steps, then the cards the tools drew).
import { useEffect, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { SPEC_DATA_PART_TYPE, type Spec } from "@json-render/core";
import { JSONUIProvider, Renderer, useJsonRenderMessage } from "@json-render/react";
import { registry } from "./registry";

// The model is asked for plain text; this drops emphasis marks if it slips.
const plain = (s: string) => s.replace(/\*\*|__/g, "").replace(/\*(?=\S)|(?<=\S)\*/g, "");
const transport = new DefaultChatTransport({ api: "/api/chat" });
const STEP: Record<string, string> = {
  show_shelves: "LOOKED AT YOUR SHELVES",
  show_claims: "LOOKED AT YOUR CLAIMS",
  search_library: "SEARCHED THE LIBRARY",
  browse_library_class: "BROWSED A SHELF OF THE LIBRARY",
  propose_claim: "PROPOSED A CLAIM",
};

/** One answer's cards. Each card keeps its own tap state, so a tap marks only its card. */
function Cards({ spec }: { spec: Spec }) {
  return (
    <JSONUIProvider registry={registry}>
      <Renderer spec={spec} registry={registry} />
    </JSONUIProvider>
  );
}

function Message({ message, working }: { message: UIMessage; working: boolean }) {
  const { spec, text } = useJsonRenderMessage(message.parts);
  if (message.role === "user") return <div className="chat-msg chat-user"><p>{text}</p></div>;
  // Parts in order: text, a run of tool steps, and the cards (once, where the first card arrived).
  type Seg = { kind: "text"; text: string } | { kind: "steps"; steps: { id: string; label: string; done: boolean }[] } | { kind: "cards" };
  const segs: Seg[] = [];
  let sawCards = false;
  for (const part of message.parts) {
    const last = segs[segs.length - 1];
    if (part.type === "text" && part.text.trim()) {
      if (last?.kind === "text") last.text += part.text;
      else segs.push({ kind: "text", text: part.text });
    } else if (part.type.startsWith("tool-")) {
      const p = part as { toolCallId: string; type: string; state: string };
      // A step that failed (the model sent input the tool refused) is left out; the model tries again.
      if (p.state === "output-error") continue;
      const step = { id: p.toolCallId, label: STEP[p.type.slice(5)] ?? p.type.slice(5).toUpperCase(), done: p.state === "output-available" };
      if (last?.kind === "steps") last.steps.push(step);
      else segs.push({ kind: "steps", steps: [step] });
    } else if (part.type === SPEC_DATA_PART_TYPE && !sawCards) {
      sawCards = true;
      segs.push({ kind: "cards" });
    }
  }
  return (
    <div className="chat-msg chat-assistant">
      {segs.map((seg, i) =>
        seg.kind === "text" ? (
          <p key={i}>{plain(seg.text)}</p>
        ) : seg.kind === "steps" ? (
          <ul key={i} className="chat-steps">{seg.steps.map((st) => <li key={st.id}>{st.done ? "✓" : "…"} {st.label}</li>)}</ul>
        ) : spec ? (
          <Cards key={i} spec={spec} />
        ) : null,
      )}
      {!segs.length && working ? <p className="nd-note">Looking in your apps...</p> : null}
    </div>
  );
}

export function Chat() {
  const [text, setText] = useState("");
  const { messages, sendMessage, status, error } = useChat({ transport });
  const busy = status === "submitted" || status === "streaming";
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  function send(e: React.FormEvent) {
    e.preventDefault();
    const said = text.trim();
    if (!said || busy) return;
    setText("");
    void sendMessage({ text: said });
  }

  return (
    <div className="nd-ask">
      <div className="chat-log" aria-live="polite">
        {messages.length ? null : <p className="nd-note">Ask about what you are learning: your shelves, a book to find, a claim you hold. I show what is in your apps, and nothing is kept unless you accept it.</p>}
        {messages.map((m, i) => <Message key={m.id} message={m} working={busy && i === messages.length - 1} />)}
        {busy && messages[messages.length - 1]?.role === "user" ? <p className="nd-note">Looking in your apps...</p> : null}
        {error ? <p className="nd-note">The model is not reachable right now, so I can't answer. Your apps are all still there.</p> : null}
        <div ref={end} />
      </div>
      <form className="chat-form" onSubmit={send}>
        <label htmlFor="chat-input" className="sr-only">Message</label>
        <input id="chat-input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Say anything..." autoComplete="off" maxLength={4000} />
        <button type="submit" className="nd-btn" disabled={busy || !text.trim()}>SEND</button>
      </form>
    </div>
  );
}
