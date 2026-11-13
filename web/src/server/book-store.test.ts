import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { beliefFromWork, bookOf, compileEdition, domainView, idealsOf, markBelief, placeClaim } from "./book-store";
import { keepClaim, setStance } from "./claims-store";
import { createWork, fileWork, getWork, saveWork } from "./work-store";

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
    await beliefFromWork(user, w, "Marriage is a covenant  before it is a contract.", "line", { kind: "ideal", origins: ["reasoning"] });
    await expect(beliefFromWork(user, w, "x", "romance", { kind: "ideal" })).rejects.toThrow(/domain/);
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
    await expect(beliefFromWork(other, w, "x", "work", { kind: "ideal" })).rejects.toThrow(/No such work/);
  });

  it("marks ideals and practical beliefs, keeps origins and holders, and files works in domains", async () => {
    const user = await newUser("Ideals Test");
    const w = await createWork(user, "essay", null, "line");
    expect((await getWork(user, w))!.domain).toBe("line");
    const ideal = await beliefFromWork(user, w, "Children are best raised by both parents.", "line", { kind: "ideal", origins: ["nature", "reasoning"], holders: [{ name: "Aristotle", citation: "Politics I.2" }] });
    await expect(beliefFromWork(user, w, "We both work part time.", "line", { kind: "practical", circumstances: "this year" })).rejects.toThrow(/ideal it serves/);
    await expect(beliefFromWork(user, w, "We both work part time.", "line", { kind: "practical", serves: { id: ideal } })).rejects.toThrow(/circumstances/);
    const p = await beliefFromWork(user, w, "We both work part time.", "line", { kind: "practical", serves: { id: ideal }, circumstances: "while the baby is under one", origins: ["appetite", "compulsion"] });
    await expect(markBelief(user, p, { kind: "practical", serves: { id: p }, circumstances: "x" })).rejects.toThrow(/not one of your ideals/);
    // A practical belief naming a new ideal keeps that ideal too, in the same domain.
    const q = await beliefFromWork(user, w, "I cook dinner on weekdays.", "line", { kind: "practical", serves: { text: "A family eats together." }, circumstances: "while I work from home" });
    const { chapter, works } = await domainView(user, "line");
    expect(works.map((x) => x.id)).toEqual([w]);
    expect(chapter.entries.map((e) => [e.text, e.kind])).toEqual([
      ["Children are best raised by both parents.", "ideal"],
      ["A family eats together.", "ideal"],
      ["We both work part time.", "practical"],
      ["I cook dinner on weekdays.", "practical"],
    ]);
    const pe = chapter.entries.find((e) => e.id === p)!;
    expect(pe).toMatchObject({ serves: { id: ideal, text: "Children are best raised by both parents." }, circumstances: "while the baby is under one", origins: ["compulsion", "appetite"] });
    expect(chapter.entries.find((e) => e.id === ideal)).toMatchObject({ holders: [{ name: "Aristotle", citation: "Politics I.2" }], origins: ["nature", "reasoning"] });
    expect(chapter.entries.find((e) => e.id === q)!.serves?.text).toBe("A family eats together.");
    expect((await idealsOf(user)).map((i) => i.text).sort()).toEqual(["A family eats together.", "Children are best raised by both parents."]);
    const raw = (await db().query("SELECT circumstances_ct FROM claim_kind WHERE user_id = $1", [user])).rows.map((r) => r.circumstances_ct ?? "").join(" ");
    expect(raw).not.toContain("baby");
    await expect(db().query("UPDATE claim_kind SET kind = 'ideal' WHERE user_id = $1", [user])).rejects.toThrow(/append-only/);
    await fileWork(user, w, null);
    expect((await getWork(user, w))!.domain).toBeNull();
    await fileWork(user, w, "household");
    expect((await domainView(user, "household")).works.map((x) => x.id)).toEqual([w]);
    await expect(fileWork(user, w, "romance")).rejects.toThrow(/domain/);
  });
});
