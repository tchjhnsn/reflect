import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { keepClaim, listClaims, setStance } from "./claims-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("claims on Neon", () => {
  afterAll(() => db().end());

  it("keeps a claim once with its cited holders, under a subject, and counts the latest stance", async () => {
    const user = await newUser("Claim Test");
    const id = await keepClaim(user, { text: "Marriage is a covenant", holders: [{ name: "Scripture", citation: "Malachi 2:14" }], subject: "Marriage", stance: "belief" });
    expect(await keepClaim(user, { text: "marriage is  a covenant", stance: "suspension" })).toBe(id);
    await setStance(user, id, "belief", "after reading Malachi");
    const [c] = await listClaims(user);
    expect(c).toMatchObject({ text: "Marriage is a covenant", holders: [{ name: "Scripture", citation: "Malachi 2:14" }], subject: { name: "Marriage" } });
    expect(c!.history.map((h) => h.stance)).toEqual(["belief", "suspension", "belief"]);
    expect(c!.stance).toMatchObject({ stance: "belief", note: "after reading Malachi" });
    const stored = (await db().query("SELECT text_ct FROM learn_claim WHERE id = $1", [id])).rows[0].text_ct as string;
    expect(stored).not.toContain("covenant");
    await expect(setStance(user, id, "hold")).rejects.toThrow();
    await expect(setStance(await newUser("Other"), id, "belief")).rejects.toThrow("no such claim");
  });
});
