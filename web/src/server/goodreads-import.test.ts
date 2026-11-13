import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { Fetch } from "@reflect/core/openlibrary";
import { importGoodreads } from "./goodreads-import";
import { learnView } from "./learn-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

const CSV = `Book Id,Title,Author,Additional Authors,ISBN,ISBN13,Date Read,Exclusive Shelf
1,Meditations,Marcus Aurelius,,"=""""","=""9780140449334""",2025/02/01,read
2,The Republic,Plato,,"=""""","=""""",,to-read
3,Unfindable Book,Nobody,,"=""""","=""""",,currently-reading
`;
const fake: Fetch = async (url) => {
  const q = decodeURIComponent(new URL(url).searchParams.get("q") ?? "");
  const docs = q.startsWith("isbn:") ? [{ key: "/works/OL1W", title: "Meditations", author_name: ["Marcus Aurelius"], lcc: ["B-0580"], cover_i: 7 }]
    : q.startsWith("(The Republic") ? [{ key: "/works/OL2W", title: "The Republic", author_name: ["Plato"], lcc: ["JC-0071"] }]
    : [{ key: "/works/OL3W", title: "Something else entirely" }];
  return { ok: true, status: 200, json: async () => ({ docs }) };
};

describe.skipIf(!live)("Goodreads import on Neon", () => {
  afterAll(() => db().end());

  it("adds each book with its shelf as status, shelved by its catalog entry, and skips books already kept", async () => {
    const user = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, 'Goodreads Test', $2, false, now(), now())`, [user, `${user}@example.test`]);
    expect(await importGoodreads(user, CSV, fake)).toEqual({ added: 3, already: 0, uncataloged: 1, overLimit: 0 });
    const books = (await learnView(user)).books;
    const by = new Map(books.map((b) => [b.title, b]));
    expect(by.get("Meditations")).toMatchObject({ status: "read", finishedOn: "2025-02-01", isbn: "9780140449334", catalog: { work: "OL1W", lcc: ["B-0580"], cover: 7 } });
    expect(by.get("The Republic")).toMatchObject({ status: "want", catalog: { work: "OL2W" } });
    expect(by.get("Unfindable Book")).toMatchObject({ status: "reading", catalog: null });
    expect(await importGoodreads(user, CSV, fake)).toEqual({ added: 0, already: 3, uncataloged: 0, overLimit: 0 });
  });
});
