"use client";

import { defineRegistry } from "@json-render/react";
import type { SetState } from "@json-render/react";
import { addBookFromChat, keepClaimFromChat } from "./actions";
import { catalog } from "./catalog";

/** Marks a card done or failed in its answer's state, which the card's props are bound to. */
const mark = (setState: SetState, key: string, ok: boolean) =>
  setState((prev) => {
    const at = ok ? "done" : "failed";
    const was = (prev[at] && typeof prev[at] === "object" ? prev[at] : {}) as Record<string, unknown>;
    return { ...prev, [at]: { ...was, [key]: true } };
  });

/** Runs a server action for one card; a failure shows on the card instead of vanishing. */
const run = async (setState: SetState, key: string, f: () => Promise<void>) => {
  try {
    await f();
    mark(setState, key, true);
  } catch {
    mark(setState, key, false);
  }
};

export const { registry, handlers } = defineRegistry(catalog, {
  components: {
    Stack: ({ children }) => <div className="chat-stack">{children}</div>,
    Shelf: ({ props }) => (
      <div className="chat-card">
        <div className="chat-card-head"><span>FROM LEARN · SHELF</span><span>{props.id}</span></div>
        <strong>{props.name}</strong>
        <p>{props.books.length ? props.books.join(", ") : "No books"}{props.subjects.length ? <span className="muted"> · {props.subjects.join(", ")}</span> : null}</p>
      </div>
    ),
    LibraryBook: ({ props, emit }) => (
      <div className="chat-row">
        <div><strong>{props.title}</strong><span className="muted"> {props.by} · {props.shelf}</span></div>
        {props.onShelves || props.done ? <span className="muted">on your shelves</span> : <span>{props.failed ? <span className="muted">Could not add. </span> : null}<button type="button" className="primary" onClick={() => emit("press")}>Add to my shelves</button></span>}
      </div>
    ),
    Claim: ({ props }) => (
      <div className="chat-row">
        <div><strong>{props.text}</strong>{props.holders ? <span className="muted"> · held by {props.holders}</span> : null}</div>
        <span className="muted">{props.stance}</span>
      </div>
    ),
    ProposeClaim: ({ props, emit }) => (
      <div className="chat-card">
        <div className="chat-card-head"><span>PROPOSED · LEARN CLAIM</span><span>{props.stance}</span></div>
        <strong>{props.text}</strong>
        <p className="muted">{[props.subject ? `under ${props.subject}` : "", props.holders ? `held by ${props.holders}` : ""].filter(Boolean).join(" · ")}</p>
        {props.done ? <span className="muted">Kept in Learn.</span> : <span>{props.failed ? <span className="muted">Could not keep it. </span> : null}<button type="button" className="primary" onClick={() => emit("press")}>Accept</button></span>}
      </div>
    ),
    Note: ({ props }) => <p className="muted">{props.text}</p>,
  },
  actions: {
    add_book: async (p, setState) => {
      if (p) await run(setState, p.key, () => addBookFromChat(p.entry));
    },
    keep_claim: async (p, setState) => {
      if (p) await run(setState, p.key, () => keepClaimFromChat(p.claim));
    },
  },
});
