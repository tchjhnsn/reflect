import { describe, expect, it } from "vitest";
import { chatPart, isChatId, readChatPart } from "./chat-journal";

describe("chat kept in the journal", () => {
  it("writes each message with who said it and reads it back", () => {
    expect(chatPart("you", " Find me a book on Aristotle ")).toBe("You: Find me a book on Aristotle");
    expect(readChatPart(chatPart("reflect", "Try the Nicomachean Ethics."))).toEqual({ who: "reflect", text: "Try the Nicomachean Ethics." });
    expect(readChatPart(chatPart("you", "Thanks"))).toEqual({ who: "you", text: "Thanks" });
  });

  it("treats a part with no speaker as the person's own note", () => {
    expect(readChatPart("Later I bought it.")).toEqual({ who: "note", text: "Later I bought it." });
  });

  it("takes only uuids as chat ids", () => {
    expect(isChatId("3f1c2a9e-8b7d-4c6e-9f0a-1b2c3d4e5f60")).toBe(true);
    expect(isChatId("abc")).toBe(false);
    expect(isChatId(undefined)).toBe(false);
  });
});
