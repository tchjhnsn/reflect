import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { confirmDeath, keepPlan, legacyStatus, planProblems, receive, stopRelease, trusteeView, type PlanInput } from "./legacy-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

const plan = (waitDays: number): PlanInput => ({
  confirmations: 2,
  waitDays,
  trustees: [{ name: "Ana", contact: "ana@example.test" }, { name: "Ben", contact: "ben@example.test" }],
  gifts: [{ to: "Sam", contact: "sam@example.test", items: [{ title: "Dear Sam", text: "Stay brave." }] }],
});

describe("legacy plan checks", () => {
  it("refuses a plan no one could carry out", () => {
    expect(planProblems({ ...plan(30), trustees: [] })).toContain("name at least one trustee who may confirm your death");
    expect(planProblems({ ...plan(30), confirmations: 3 })).toContain("the plan asks 3 confirmations of 2 trustee(s)");
    expect(planProblems({ ...plan(30), gifts: [{ to: "Sam", contact: "s", items: [] }] })).toContain("the gift to Sam has nothing in it");
  });
});

describe.skipIf(!live)("legacy release on Neon", () => {
  afterAll(() => db().end());

  it("needs enough trustees, then the wait, and shows a recipient nothing before", async () => {
    const user = await newUser("Torian Test");
    const { trustees } = await keepPlan(user, plan(0));
    const [ana, ben] = trustees.map((t) => t.token);
    const status = await legacyStatus(user);
    expect(status.trustees.map((t) => t.name)).toEqual(["Ana", "Ben"]);
    expect(status.release).toEqual({ state: "none" });
    const samToken = (await db().query("SELECT token_ct FROM legacy_gift WHERE user_id = $1", [user])).rows.length;
    expect(samToken).toBe(1);

    expect(await confirmDeath(ana!)).toEqual({ state: "confirming", confirmed: 1, needed: 2 });
    expect(await confirmDeath(ana!)).toEqual({ state: "confirming", confirmed: 1, needed: 2 });
    const before = await trusteeView(ben!);
    expect(before).toMatchObject({ person: "Torian Test", trustee: "Ben", confirmedByYou: false, recipients: [] });
    expect((await confirmDeath(ben!)).state).toBe("released");
    const view = await trusteeView(ana!);
    expect(view!.recipients.map((r) => [r.to, r.contact])).toEqual([["Sam", "sam@example.test"]]);
    expect(await receive(view!.recipients[0]!.token)).toEqual({ from: "Torian Test", items: [{ title: "Dear Sam", text: "Stay brave." }] });
    await expect(stopRelease({ trusteeToken: ana! })).rejects.toThrow(/already happened/);
    await expect(keepPlan(user, plan(0))).rejects.toThrow(/release is under way/);
  });

  it("waits, and the person or a trustee can stop it; a new plan revokes old links", async () => {
    const user = await newUser("Wait Test");
    const { trustees } = await keepPlan(user, plan(30));
    const [ana, ben] = trustees.map((t) => t.token);
    await confirmDeath(ana!);
    const waiting = await confirmDeath(ben!);
    expect(waiting).toMatchObject({ state: "waiting", confirmed: 2, needed: 2 });
    const gift = (await legacyStatus(user)).gifts;
    expect(gift).toEqual([{ to: "Sam", contact: "sam@example.test" }]);
    const view = await trusteeView(ana!);
    expect(view!.recipients).toEqual([]);
    await stopRelease({ userId: user });
    expect((await legacyStatus(user)).release).toEqual({ state: "none" });
    // A false start by one trustee, stopped by another.
    await confirmDeath(ana!);
    await stopRelease({ trusteeToken: ben! });
    expect((await legacyStatus(user)).release).toEqual({ state: "none" });
    const next = await keepPlan(user, { ...plan(30), confirmations: 1, trustees: [{ name: "Cy", contact: "cy@example.test" }] });
    expect(await trusteeView(ana!)).toBeNull();
    await expect(confirmDeath(ana!)).rejects.toThrow(/not valid/);
    expect((await confirmDeath(next.trustees[0]!.token)).state).toBe("waiting");
    expect(await receive("not-a-token")).toBeNull();
  });
});
