"use client";

// The chat with AI Elements (shadcn): Conversation, Message, Tool, Suggestion and Prompt Input, over the
// same /api/chat and the same cards. A preview to compare with /chat.
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

const transport = new DefaultChatTransport({ api: "/api/chat" });
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

export function ShadcnChat() {
  const { messages, sendMessage, status } = useChat({ transport });
  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col p-6">
      <div className="mb-4 flex items-center justify-between">
        <a href="/" className="text-muted-foreground text-sm hover:text-foreground">← Home</a>
        <span className="text-sm font-medium">Ask</span>
        <a href="/chat" className="text-muted-foreground text-sm hover:text-foreground">Current look</a>
      </div>
      <Conversation className="flex-1">
        <ConversationContent>
          {messages.length === 0 ? (
            <ConversationEmptyState title="Ask about what you are learning" description="Your shelves, a book to find, a claim you hold. Nothing is kept unless you accept it." />
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
