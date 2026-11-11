import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { GUIDE, toolAddBook, toolLearning, toolRecallCheck, toolGetEntry, toolListEntries, toolWriteEntry } from "./mcp-tools";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

describe("the guide an outside AI reads", () => {
  it("says the person decides and names no element model", () => {
    expect(GUIDE).toContain("save only what the person asked you to save");
    expect(GUIDE).not.toMatch(/element|binding/i);
  });
});

describe.skipIf(!live)("MCP tools on Neon", () => {
  afterAll(() => db().end());

  it("writes only today's entry and answers for one person only", async () => {
    const user = randomUUID();
    const other = randomUUID();
    for (const id of [user, other]) {
      await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
    }
    const now = new Date("2026-11-10T12:00:00Z");
    await expect(toolWriteEntry(user, "Old news.", "2020-01-01", now)).rejects.toThrow(/today/);
    const { id } = await toolWriteEntry(user, "Walked to the lake with Mom, slowly.", "2026-11-10", now);

    expect((await toolListEntries(user, 5)).map((e) => e.id)).toEqual([id]);
    expect((await toolGetEntry(user, id)).text).toBe("Walked to the lake with Mom, slowly.");
    await expect(toolGetEntry(other, id)).rejects.toThrow(/no such entry/);
    expect(await toolListEntries(other, 5)).toEqual([]);
  });

  it("keeps a book from the person's AI, with Open Library's subjects only suggested, and a recall check", async () => {
    const user = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [user, `${user}@example.test`]);
    const fetcher = async (url: string) => ({ ok: true, status: 200, json: async () => (url.includes("search.json") ? { docs: [{ key: "/works/OL1W", title: "Meditations", author_name: ["Marcus Aurelius"], subject: ["Stoics", "Ethics"] }] } : { table_of_contents: ["Book One"] }) });
    const added = await toolAddBook(user, { title: "Meditations", isbn: "9780140449334", status: "read", subjects: ["Stoicism"] }, fetcher);
    expect(added.authors).toEqual(["Marcus Aurelius"]);
    expect(added.suggested_subjects).toEqual(["Stoics", "Ethics"]);
    const r = await toolRecallCheck(user, added.book_id, "Mostly stoicism and duty.");
    expect(r.named).toEqual(["Stoicism"]);
    expect(r.contents).toEqual(["Book One"]);
    const l = await toolLearning(user);
    expect(l.known).toEqual([]);
    expect(l.books[0]!.subjects).toEqual(["Stoicism"]);
    expect(l.books[0]!.not_yet_rated).toEqual(["Stoicism"]);
  });
});
