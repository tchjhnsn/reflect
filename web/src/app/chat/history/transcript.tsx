import type { Entry } from "@reflect/core/journal";
import { readChatPart } from "@reflect/core/chat-journal";
import { cn } from "@/lib/utils";

/** A kept chat, read back: the person's messages on the right, Reflect's on the left, notes dashed. */
export function Transcript({ entry }: { entry: Entry }) {
  return (
    <div className="space-y-3">
      {entry.parts.map((p) => {
        const m = readChatPart(p.text);
        return (
          <div key={p.seq} className={cn("flex", m.who === "you" ? "justify-end" : "justify-start")}>
            <p className={cn("max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap", m.who === "you" ? "bg-secondary" : m.who === "note" ? "border border-dashed" : "")}>
              {m.text}
            </p>
          </div>
        );
      })}
    </div>
  );
}
