import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { bookOf, compileEdition, holdFromWork, placeClaim } from "./book-store";
import { keepClaim, setStance } from "./claims-store";
import { createWork, getWork, saveWork } from "./work-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("the book on Neon", () => {
  afterAll(() => db().end());

  it("compiles beliefs marked in works and placed claims into chapters, and keeps an edition as a work", async () => {
    const user = await newUser("Book Test");
    const w = await createWork(user, "essay");
    await saveWork(user, w, "On marriage", "Marriage is a covenant before it is a contract.");
    await holdFromWork(user, w, "Marriage is a covenant  before it is a contract.", "line");
    await expect(holdFromWork(user, w, "x", "romance")).rejects.toThrow(/chapter/);
    const c = await keepClaim(user, { text: "Work is only for money.", stance: "belief" });
    await placeClaim(user, c, "work");
    await setStance(user, c, "disbelief");
    await keepClaim(user, { text: "Unplaced belief.", stance: "belief" });
    const { chapters, unplaced } = await bookOf(user);
    expect(chapters.find((x) => x.domain === "line")!.entries[0]).toMatchObject({ text: "Marriage is a covenant before it is a contract.", state: "held", source: { workId: w, title: "On marriage" } });
    expect(chapters.find((x) => x.domain === "work")!.entries[0]).toMatchObject({ state: "once" });
    expect(unplaced.map((e) => e.text)).toEqual(["Unplaced belief."]);
    const ed = (await getWork(user, await compileEdition(user)))!;
    expect(ed.kind).toBe("memoir");
    expect(ed.body).toContain("## Line");
    expect(ed.body).toContain("~~Work is only for money.~~");
    const other = await newUser("Other");
    await expect(placeClaim(other, c, "work")).rejects.toThrow(/No such belief/);
    await expect(holdFromWork(other, w, "x", "work")).rejects.toThrow(/No such work/);
  });
});
