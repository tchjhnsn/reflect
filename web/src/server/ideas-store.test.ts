import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { readSweep } from "@reflect/core/sweep";
import { db } from "@/db/pool";
import { addBook, learnView } from "./learn-store";
import { saveUpload, getUpload } from "./learn-sources-store";
import { addInput, getIdea, keepIdea, listIdeas, listInputs, markSubject, resolveByTitle } from "./ideas-store";
import { keepSweepItems } from "./sweep-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

describe.skipIf(!live)("inputs, subject marks, ideas and the memory sweep on Neon", () => {
  afterAll(() => db().end());

  it("keeps inputs once, marks that are never grasps, and ideas whose latest line decides each input", async () => {
    const user = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [user, `${user}@example.test`]);
    const book = await addBook(user, { title: "Politics", authors: ["Aristotle"], status: "referenced" });
    const film = await addInput(user, { kind: "film", title: "Metropolis", film: "film", year: 1927, status: "watched" }, ["Urban design"]);
    expect(await addInput(user, { kind: "film", title: "metropolis", film: "film", status: "want" })).toBe(film);
    const link = await addInput(user, { kind: "link", title: "Zoning thread", link: "thread", url: "https://www.reddit.com/r/urbanplanning/comments/abc" });
    await expect(addInput(user, { kind: "talk", title: "Strong Towns", speaker: " " })).rejects.toThrow(/who spoke/);
    expect((await listInputs(user)).find((i) => i.id === link)!.body).toMatchObject({ source: "reddit" });

    await markSubject(user, "Political philosophy", "shown", "Claude", "explained the six constitutions");
    await markSubject(user, "Urban studies", "suggested", "Claude", "your interest in cities");
    const v = await learnView(user);
    expect(v.known).toEqual([]);
    expect(v.shown.map((s) => s.name)).toEqual(["Political philosophy"]);
    expect(v.suggested.map((s) => s.name)).toEqual(["Urban studies"]);
    await expect(db().query("DELETE FROM learn_subject_mark WHERE user_id = $1", [user])).rejects.toThrow(/append-only/);

    await expect(keepIdea(user, "Corrupt cities", {})).rejects.toThrow(/statement/);
    const id = await keepIdea(user, "Corrupt and correct cities", { statement: "Cities in correct and corrupt forms.", from: [{ target: { ref: `book:${book}` }, how: "Politics III" }], maybe: [{ target: { ref: `input:${film}` } }, { target: { text: "a sermon" } }] });
    await keepIdea(user, "corrupt and correct cities", { related: [{ target: { text: "The Death and Life of Great American Cities" }, stance: "challenges", by: "Claude" }], from: [{ target: { ref: `input:${film}` } }] });
    await expect(keepIdea(user, "Corrupt and correct cities", { from: [{ target: { ref: `idea:${id}` } }] })).rejects.toThrow(/own input/);
    await expect(keepIdea(user, "Corrupt and correct cities", { from: [{ target: { ref: `book:${randomUUID()}` } }] })).rejects.toThrow(/not yours/);
    const i = (await getIdea(user, id))!;
    expect(i.statement).toBe("Cities in correct and corrupt forms.");
    expect(i.from.map((x) => i.labels[x.ref!] ?? x.text)).toEqual(["Politics", "Metropolis"]);
    expect(i.maybe.map((x) => x.text)).toEqual(["a sermon"]);
    expect(i.related.map((r) => [r.stance, r.text, r.by])).toEqual([["challenges", "The Death and Life of Great American Cities", "Claude"]]);
    await keepIdea(user, "Corrupt and correct cities", { drop: [{ text: "a sermon" }] });
    expect((await getIdea(user, id))!.maybe).toEqual([]);
    expect(await resolveByTitle(user, "metropolis")).toEqual({ ref: `input:${film}` });
    await expect(db().query("UPDATE idea_line SET kind = 'from' WHERE user_id = $1", [user])).rejects.toThrow(/append-only/);
  });

  it("keeps the items of a memory sweep the person chose, once, with inferred inputs as maybe from", async () => {
    const user = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [user, `${user}@example.test`]);
    const s = readSweep(JSON.stringify({
      kind: "reflect.learn-source",
      version: 1,
      reported_by: { assistant: "Claude", reported_at: "2026-11-11" },
      books: [{ title: "Politics", authors: ["Aristotle"], engaged: "referenced", basis: "i-said" }],
      subjects: { shown: [{ name: "Political philosophy", how: "explained", evidence: "the six constitutions" }], suggested: [{ name: "Urban studies" }] },
      ideas: [{ title: "Corrupt and correct cities", statement: "Aristotle's forms applied to cities.", inputs: [{ title: "Politics" }], basis: "you-inferred" }],
    }));
    const up = await saveUpload(user, "sweep", "reply.md", s);
    expect(await keepSweepItems(user, up, [3, 0, 1])).toBe(3);
    expect(await keepSweepItems(user, up, [0])).toBe(0);
    expect((await getUpload(user, up))!.kept.sort()).toEqual([0, 1, 3]);
    const [idea] = await listIdeas(user);
    expect(idea!.maybe.map((x) => idea!.labels[x.ref!])).toEqual(["Politics"]);
    const v = await learnView(user);
    expect(v.books.map((b) => b.status)).toEqual(["referenced"]);
    expect(v.shown.map((x) => x.name)).toEqual(["Political philosophy"]);
    expect(v.suggested).toEqual([]);
  });
});
