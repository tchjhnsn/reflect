"use client";

// The same cards as ../../chat/registry.tsx, drawn with shadcn instead of the app's own CSS.
import { useState } from "react";
import { defineRegistry } from "@json-render/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { addBookFromChat, keepClaimFromChat } from "./actions";
import { catalog } from "./catalog";

function Tap({ label, doneText, already, run }: { label: string; doneText: string; already?: boolean; run: () => Promise<void> }) {
  const [tap, setTap] = useState<"idle" | "busy" | "done" | "failed">(already ? "done" : "idle");
  if (tap === "done") return <Badge variant="secondary">{doneText}</Badge>;
  return (
    <div className="flex items-center gap-2">
      {tap === "failed" ? <span className="text-destructive text-xs">Didn't go through.</span> : null}
      <Button
        size="sm"
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
      </Button>
    </div>
  );
}

export const { registry } = defineRegistry(catalog, {
  components: {
    Stack: ({ children }) => <div className="mt-3 flex flex-col gap-2">{children}</div>,
    Shelf: ({ props }) => (
      <div className="rounded-lg border bg-card p-4 shadow-xs">
        <div className="text-muted-foreground mb-1 flex justify-between text-xs"><span>Shelf</span><span>{props.id}</span></div>
        <div className="font-medium">{props.name}</div>
        <p className="text-muted-foreground mt-1 text-sm">{[...props.books, ...props.subjects].join(", ") || "Empty"}</p>
      </div>
    ),
    LibraryBook: ({ props }) => (
      <div className="flex items-center justify-between gap-4 rounded-lg border bg-card p-3 shadow-xs">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">{props.title}</div>
          <div className="text-muted-foreground truncate text-xs">{props.by} · {props.shelf}</div>
        </div>
        <Tap label="Add to shelves" doneText="On your shelves" already={props.onShelves} run={() => addBookFromChat(props.entry)} />
      </div>
    ),
    Claim: ({ props }) => (
      <div className="flex items-center justify-between gap-4 rounded-lg border bg-card p-3 shadow-xs">
        <div className="min-w-0">
          <div className="text-sm font-medium">{props.text}</div>
          {props.holders ? <div className="text-muted-foreground text-xs">Held by {props.holders}</div> : null}
        </div>
        <Badge variant="outline">{props.stance}</Badge>
      </div>
    ),
    ProposeClaim: ({ props }) => (
      <div className="rounded-lg border bg-card p-4 shadow-xs">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-muted-foreground text-xs">Proposed claim</span>
          <Badge variant="outline">{props.stance}</Badge>
        </div>
        <div className="font-medium">{props.text}</div>
        <p className="text-muted-foreground mt-1 mb-3 text-sm">{[props.subject ? `Under ${props.subject}` : "", props.holders ? `held by ${props.holders}` : ""].filter(Boolean).join(" · ")}</p>
        <Tap label="Accept" doneText="Kept in Learn" run={() => keepClaimFromChat(props.claim)} />
      </div>
    ),
    Note: ({ props }) => <p className="text-muted-foreground text-sm">{props.text}</p>,
  },
});
