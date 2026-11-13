import { describe, expect, it } from "vitest";
import { librivoxBook } from "./librivox.ts";
import type { Fetch } from "./openlibrary.ts";

describe("librivox", () => {
  it("reads the first English recording with its chapters, over https from the Internet Archive only", async () => {
    const seen: string[] = [];
    const books: Record<string, unknown> = {
      "5349": { books: [{ id: "5349", title: "Orgullo y prejuicio", language: "Spanish", sections: [{ listen_url: "http://www.archive.org/download/x/1.mp3" }] }] },
      "253": { books: [{ id: "253", title: "Pride and Prejudice", language: "English", totaltime: "13:06:44", url_librivox: "https://librivox.org/pride-and-prejudice-by-jane-austen/", sections: [
        { section_number: "1", title: "Chapters 1-3", listen_url: "http://www.archive.org/download/pp/01.mp3", playtime: "1132", readers: [{ display_name: "Chris Goringe" }] },
        { section_number: "2", title: "Elsewhere", listen_url: "https://evil.example/2.mp3", playtime: "10" },
      ] }] },
    };
    const f: Fetch = async (url) => {
      seen.push(url);
      const id = /id=(\d+)/.exec(url)?.[1] ?? "";
      return { ok: true, status: 200, json: async () => books[id] ?? { books: [] } };
    };
    const a = await librivoxBook(["5349", "orgullo", "253"], f);
    expect(a).toEqual({ id: "253", title: "Pride and Prejudice", url: "https://librivox.org/pride-and-prejudice-by-jane-austen/", seconds: 47204, chapters: [{ n: 1, title: "Chapters 1-3", url: "https://www.archive.org/download/pp/01.mp3", seconds: 1132, reader: "Chris Goringe" }] });
    expect(seen).toHaveLength(2);
    expect(await librivoxBook([], f)).toBeNull();
  });
});
