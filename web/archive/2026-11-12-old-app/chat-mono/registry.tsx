"use client";

import { useState } from "react";
import { defineRegistry } from "@json-render/react";
import { addBookFromChat, keepClaimFromChat } from "./actions";
import { catalog } from "./catalog";

type Tap = "idle" | "busy" | "done" | "failed";

/** One tap that writes: Add to shelves or Accept. The card itself shows busy, done or failed. */
function TapButton({ label, doneText, failText, already, run }: { label: string; doneText: string; failText: string; already?: boolean; run: () => Promise<void> }) {
  const [tap, setTap] = useState<Tap>(already ? "done" : "idle");
  if (tap === "done") return <span className="muted">{doneText}</span>;
  return (
    <span>
      {tap === "failed" ? <span className="muted">{failText} </span> : null}
      <button
        type="button"
        className="nd-btn"
        disabled={tap === "busy"}
        onClick={async () => {
          setTap("busy");
          try {
            await run();
            setTap("done");
          } catch {
            setTap("failed");
          }
        }}
      >
        {label}
      </button>
    </span>
  );
}

export const { registry } = defineRegistry(catalog, {
  components: {
    Stack: ({ children }) => <div className="chat-stack">{children}</div>,
    Shelf: ({ props }) => (
      <div className="chat-card">
        <div className="chat-card-head"><span>FROM LEARN · SHELF</span><span>{props.id}</span></div>
        <strong>{props.name}</strong>
        <p>{props.books.length ? props.books.join(", ") : "No books"}{props.subjects.length ? <span className="muted"> · {props.subjects.join(", ")}</span> : null}</p>
      </div>
    ),
    LibraryBook: ({ props }) => (
      <div className="chat-row">
        <div><strong>{props.title}</strong><span className="muted"> {props.by} · {props.shelf}</span></div>
        <TapButton label="ADD TO SHELVES" doneText="on your shelves" failText="Could not add." already={props.onShelves} run={() => addBookFromChat(props.entry)} />
      </div>
    ),
    Claim: ({ props }) => (
      <div className="chat-row">
        <div><strong>{props.text}</strong>{props.holders ? <span className="muted"> · held by {props.holders}</span> : null}</div>
        <span className="muted">{props.stance}</span>
      </div>
    ),
    ProposeClaim: ({ props }) => (
      <div className="chat-card">
        <div className="chat-card-head"><span>PROPOSED · LEARN CLAIM</span><span>{props.stance}</span></div>
        <strong>{props.text}</strong>
        <p className="muted">{[props.subject ? `under ${props.subject}` : "", props.holders ? `held by ${props.holders}` : ""].filter(Boolean).join(" · ")}</p>
        <TapButton label="ACCEPT" doneText="Kept in Learn." failText="Could not keep it." run={() => keepClaimFromChat(props.claim)} />
      </div>
    ),
    Note: ({ props }) => <p className="muted">{props.text}</p>,
  },
});
