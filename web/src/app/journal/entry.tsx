"use client";

import { useState, useTransition } from "react";
import type { Entry } from "@reflect/core/journal";
import { CHAT_PROMPT, readChatPart } from "@reflect/core/chat-journal";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { closeEntryAction } from "./actions";
import { Write } from "./write";

const day = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });
const time = (iso: string) => new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });

/** One entry: its parts in order, and while open, a way to add to it or close it. */
export function EntryCard({ entry }: { entry: Entry }) {
  const [adding, setAdding] = useState(false);
  const [pending, start] = useTransition();
  const open = !entry.closedAt;
  const chat = entry.promptId === CHAT_PROMPT;
  return (
    <Card className="gap-3 py-5">
      <CardHeader className="px-5">
        <CardTitle className="flex items-center gap-2 text-sm">
          {day(entry.localDate)}
          {chat ? <Badge variant="outline">Chat with Reflect</Badge> : null}
          {open && !chat ? <Badge variant="secondary">Open</Badge> : null}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-5">
        {chat
          ? entry.parts.map((p) => {
              const m = readChatPart(p.text);
              return (
                <div key={p.seq} className={cn("flex", m.who === "you" ? "justify-end" : "justify-start")}>
                  <p className={cn("max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap", m.who === "you" ? "bg-secondary" : m.who === "note" ? "border border-dashed" : "")}>
                    {m.text}
                  </p>
                </div>
              );
            })
          : entry.parts.map((p) => (
              <div key={p.seq} className="space-y-1">
                <p className="text-muted-foreground text-xs">{time(p.createdAt)}</p>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{p.text}</p>
              </div>
            ))}
        {chat && entry.parts[0] ? <p className="text-muted-foreground text-xs">{time(entry.parts[0].createdAt)} to {time(entry.parts.at(-1)!.createdAt)}</p> : null}
        {open && !chat ? (
          adding ? (
            <Write entryId={entry.id} placeholder="Add to this entry..." onDone={() => setAdding(false)} />
          ) : (
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setAdding(true)}>Add to it</Button>
              <Button size="sm" variant="ghost" disabled={pending} onClick={() => start(async () => void (await closeEntryAction(entry.id)))}>Close entry</Button>
            </div>
          )
        ) : null}
      </CardContent>
    </Card>
  );
}
