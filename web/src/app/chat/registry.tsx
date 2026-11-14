"use client";

// The same cards as ../../chat/registry.tsx, drawn with shadcn instead of the app's own CSS.
import { useState } from "react";
import Link from "next/link";
import { ArrowRightIcon } from "lucide-react";
import { defineRegistry } from "@json-render/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { addBookFromChat, keepClaimFromChat, addItemFromChat } from "./actions";
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
    FoundItem: ({ props }) => (
      <div className="flex items-center gap-3 rounded-lg border bg-card p-3 shadow-xs">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {props.image ? <img src={props.image} alt="" className="h-14 w-14 shrink-0 rounded object-cover" /> : null}
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">{props.title}</div>
          <div className="text-muted-foreground text-xs">{[props.medium, props.by, props.source].filter(Boolean).join(" · ")}</div>
        </div>
        {props.kept ? <Badge variant="secondary">On your shelves</Badge> : <Tap label="Add" doneText="Added" run={() => addItemFromChat(props.item)} />}
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
    BrowseLink: ({ props }) => (
      <Button asChild variant="ghost" size="sm" className="w-fit">
        <Link href={props.href}>{props.label}<ArrowRightIcon /></Link>
      </Button>
    ),
    Quote: ({ props }) => (
      <Link href={props.href} className="hover:bg-muted block rounded-lg border bg-card p-4 shadow-xs">
        <blockquote className="font-serif">{props.text}</blockquote>
        <p className="text-muted-foreground mt-2 text-xs">{props.cite}</p>
      </Link>
    ),
    Note: ({ props }) => <p className="text-muted-foreground text-sm">{props.text}</p>,
  },
});
