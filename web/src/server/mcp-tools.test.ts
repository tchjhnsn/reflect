import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { PRIMES } from "@reflect/core/state";
import { GUIDE, toolConceptHistory, toolGetEntry, toolKeepReading, toolListEntries, toolWriteEntry } from "./mcp-tools";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

describe("the guide an outside AI reads", () => {
  it("names every element and the rule that the person decides", () => {
    for (const p of PRIMES) expect(GUIDE).toContain(p);
    expect(GUIDE).toContain("only with what they said yes to");
  });
});

describe.skipIf(!live)("MCP tools on Neon", () => {
  afterAll(() => db().end());

  it("writes only today's entry, keeps a reading, and answers for one person only", async () => {
    const user = randomUUID();
    const other = randomUUID();
    for (const id of [user, other]) {
      await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
    }
    const now = new Date("2026-11-10T12:00:00Z");
    await expect(toolWriteEntry(user, "Old news.", "2020-01-01", now)).rejects.toThrow(/today/);
    const { id } = await toolWriteEntry(user, "Walked to the lake with Mom, slowly.", "2026-11-10", now);
    await toolKeepReading(user, id, [{ element: "attachment", weight: 0.9, on: [{ concept: { kind: "entity", name: "Mom", sort: "person" }, words: "Mom" }, { concept: { kind: "quality", name: "slow" }, words: "slowly", role: "manner" }] }]);

    expect((await toolListEntries(user, 5)).map((e) => e.id)).toEqual([id]);
    expect((await toolGetEntry(user, id)).readings).toHaveLength(1);
    await expect(toolGetEntry(other, id)).rejects.toThrow(/no such entry/);
    expect(await toolListEntries(other, 5)).toEqual([]);
    const mom = await toolConceptHistory(user, { kind: "entity", name: "mom", sort: "person" });
    expect(mom.entries).toEqual([id]);
    expect(mom.now.attachment!.value).toBeCloseTo(0.9, 5);
  });
});
