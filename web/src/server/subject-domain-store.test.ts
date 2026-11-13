import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { bookOf, claimFromWork, claimWorks } from "./book-store";
import { createWork, saveWork } from "./work-store";
import { keepClaim } from "./claims-store";
import { learnIn, placeSubject, subjectDomains, subjectNames } from "./subject-domain-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("subjects placed in domains on Neon", () => {
  afterAll(() => db().end());

  it("places a Learn subject once and its claims show in that domain and its chapter of the book", async () => {
    const user = await newUser("Subject Domain Test");
    const c = await keepClaim(user, { text: "Marriage is a covenant.", stance: "belief", subject: "Marriage" });
    await keepClaim(user, { text: "Divorce is never allowed.", stance: "suspension", subject: "Marriage" });
    await keepClaim(user, { text: "Rent beats buying.", stance: "belief", subject: "Housing" });
    const before = await learnIn(user, "line");
    expect(before.subjects).toEqual([]);
    expect(before.unplaced.map((s) => s.name)).toEqual(["Housing", "Marriage"]);
    const marriage = before.unplaced.find((s) => s.name === "Marriage")!.id;
    await placeSubject(user, marriage, "line");
    const after = await learnIn(user, "line");
    expect(after.subjects.map((s) => [s.name, s.claims.map((x) => [x.text, x.stance])])).toEqual([["Marriage", [["Marriage is a covenant.", "belief"], ["Divorce is never allowed.", "suspension"]]]]);
    expect(after.unplaced.map((s) => s.name)).toEqual(["Housing"]);
    const { chapters, unplaced } = await bookOf(user);
    expect(chapters.find((x) => x.domain === "line")!.entries.map((e) => e.id)).toEqual([c]);
    expect(unplaced.map((e) => e.text)).toEqual(["Rent beats buying."]);
    await placeSubject(user, marriage, null);
    expect((await subjectDomains(user)).has(marriage)).toBe(false);
    await expect(placeSubject(user, marriage, "romance")).rejects.toThrow(/domain/);
    await expect(placeSubject(await newUser("Other"), marriage, "line")).rejects.toThrow(/No such subject/);
    await expect(db().query("UPDATE subject_domain SET domain = 'work' WHERE user_id = $1", [user])).rejects.toThrow(/append-only/);
  });

  it("sends a passage from a work to Learn as a claim under a subject, which reaches the subject's domain", async () => {
    const user = await newUser("Send To Learn Test");
    const w = await createWork(user, "blank");
    await saveWork(user, w, "Thursday", "A family eats together.");
    await keepClaim(user, { text: "Kids need a routine.", stance: "belief", subject: "Raising children" });
    const sid = (await learnIn(user, "line")).unplaced[0]!.id;
    await placeSubject(user, sid, "line");
    const c = await claimFromWork(user, w, " A family  eats together. ", "Raising children", "belief");
    await expect(claimFromWork(user, w, "x", "  ", "belief")).rejects.toThrow(/subject/);
    await expect(claimFromWork(user, w, "x", "Raising children", "hope")).rejects.toThrow(/stance/);
    await expect(claimFromWork(await newUser("Other"), w, "x", "Raising children", "belief")).rejects.toThrow(/No such work/);
    expect(await subjectNames(user)).toEqual(["Raising children"]);
    expect((await claimWorks(user)).get(c)).toEqual({ workId: w, title: "Thursday" });
    const line = (await learnIn(user, "line")).subjects[0]!;
    expect(line.claims.map((x) => x.text)).toEqual(["Kids need a routine.", "A family eats together."]);
    expect((await bookOf(user)).chapters.find((x) => x.domain === "line")!.entries.find((e) => e.id === c)?.source).toEqual({ workId: w, title: "Thursday" });
  });
});
