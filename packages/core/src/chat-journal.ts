// A chat kept in the journal (docs/design/REFLECT_UI_2026-11-12.md section 5): one entry per chat, marked by
// its prompt id, with one part per message, each starting with who said it. Only the words are kept; cards
// the person accepted are already items in their app.

/** The prompt id that marks a journal entry as a chat. */
export const CHAT_PROMPT = "chat";

export type Speaker = "you" | "reflect";
const LABEL: Record<Speaker, string> = { you: "You", reflect: "Reflect" };

/** One message as a journal part: "You: ..." or "Reflect: ...". */
export function chatPart(who: Speaker, text: string): string {
  return `${LABEL[who]}: ${text.trim()}`;
}

/** Read a part back. A part without a speaker (the person added it in the journal) is the person's own note. */
export function readChatPart(part: string): { who: Speaker | "note"; text: string } {
  if (part.startsWith("You: ")) return { who: "you", text: part.slice(5) };
  if (part.startsWith("Reflect: ")) return { who: "reflect", text: part.slice(9) };
  return { who: "note", text: part };
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
/** A chat's id is its journal entry's id, so it must be a uuid. */
export const isChatId = (id: unknown): id is string => typeof id === "string" && UUID.test(id);
