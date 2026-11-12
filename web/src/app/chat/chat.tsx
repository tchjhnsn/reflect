"use client";

// The chat with AI Elements (shadcn): Conversation, Message, Tool, Suggestion and Prompt Input, over the
// same /api/chat and the same cards. Torian chose this look over the mono one (2026-11-12).
import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type ToolUIPart, type UIMessage } from "ai";
import { SPEC_DATA_PART_TYPE } from "@json-render/core";
import { JSONUIProvider, Renderer, useJsonRenderMessage } from "@json-render/react";
import { Conversation, ConversationContent, ConversationEmptyState, ConversationScrollButton } from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import { PromptInput, PromptInputBody, PromptInputFooter, PromptInputSubmit, PromptInputTextarea, PromptInputTools } from "@/components/ai-elements/prompt-input";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { Tool, ToolHeader } from "@/components/ai-elements/tool";
import { registry } from "./registry";

const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};
// The device's date goes with each message, so the journal entry this chat is kept in has the person's day.
const transport = new DefaultChatTransport({ api: "/api/chat", body: () => ({ localDate: today() }) });
const TITLE: Record<string, string> = {
  show_shelves: "Looked at your shelves",
  show_claims: "Looked at your claims",
  search_library: "Searched the library",
  browse_library_class: "Browsed a shelf of the library",
  propose_claim: "Proposed a claim",
};
const STARTERS = ["What's on my shelves?", "Find me a good introduction to Aristotle", "Show my claims"];

function Answer({ message }: { message: UIMessage }) {
  const { spec } = useJsonRenderMessage(message.parts);
  let cardsShown = false;
  return (
    <>
      {message.parts.map((part, i) => {
        if (part.type === "text" && part.text.trim()) return <MessageResponse key={i}>{part.text}</MessageResponse>;
        if (part.type.startsWith("tool-")) {
          const t = part as ToolUIPart;
          if (t.state === "output-error") return null;
          return (
            <Tool key={i}>
              <ToolHeader type={t.type} state={t.state} title={TITLE[t.type.slice(5)] ?? t.type.slice(5)} />
            </Tool>
          );
        }
        if (part.type === SPEC_DATA_PART_TYPE && spec && !cardsShown) {
          cardsShown = true;
          return (
            <JSONUIProvider key={i} registry={registry}>
              <Renderer spec={spec} registry={registry} />
            </JSONUIProvider>
          );
        }
        return null;
      })}
    </>
  );
}

export function Chat() {
  // A chat's id is its journal entry's id (a uuid), made once per visit to the page.
  const [id] = useState(() => crypto.randomUUID());
  const { messages, sendMessage, status } = useChat({ id, transport });
  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col p-6">
      <div className="mb-4 flex items-center justify-between">
        <a href="/" className="text-muted-foreground text-sm hover:text-foreground">← Home</a>
        <span className="text-sm font-medium">Ask</span>
        <span className="w-12" aria-hidden="true" />
      </div>
      <Conversation className="flex-1">
        <ConversationContent>
          {messages.length === 0 ? (
            <ConversationEmptyState title="Ask about what you are learning" description="Your shelves, a book to find, a claim you hold. This chat is kept in your journal; nothing goes into your apps unless you accept it." />
          ) : (
            messages.map((m) => (
              <Message key={m.id} from={m.role}>
                <MessageContent>{m.role === "user" ? m.parts.map((p, i) => (p.type === "text" ? <span key={i}>{p.text}</span> : null)) : <Answer message={m} />}</MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
      {messages.length === 0 ? (
        <Suggestions className="mb-3">
          {STARTERS.map((s) => <Suggestion key={s} suggestion={s} onClick={(t) => void sendMessage({ text: t })} />)}
        </Suggestions>
      ) : null}
      <PromptInput onSubmit={(m) => { if (m.text?.trim()) void sendMessage({ text: m.text }); }}>
        <PromptInputBody>
          <PromptInputTextarea placeholder="Say anything..." />
        </PromptInputBody>
        <PromptInputFooter>
          <PromptInputTools />
          <PromptInputSubmit status={status} />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
}
