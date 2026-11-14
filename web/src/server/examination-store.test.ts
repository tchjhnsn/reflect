import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { attest, closeExamination, createExamination, examinationByToken, getExamination, joinExamination, listExaminations } from "./examination-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("examinations on Neon", () => {
  afterAll(() => db().end());

  it("puts a person under examination, lets invited people examine and attest once, and keeps the record", async () => {
    const [ann, ben, cal] = [await newUser("Ann"), await newUser("Ben"), await newUser("Cal")];
    await expect(createExamination(ann, { courseUrl: "https://example.com/x", courseTitle: "X", name: "Ann", claim: "y" })).rejects.toThrow();
    const { id, token } = await createExamination(ann, { courseUrl: "https://oyc.yale.edu/philosophy/phil-176", courseTitle: "Death", name: " Ann R. ", claim: "Why death is bad for the one who dies." });
    const sealed = (await db().query("SELECT claim_ct FROM examination WHERE id = $1", [id])).rows[0].claim_ct as string;
    expect(sealed).not.toContain("death");
    expect(await joinExamination(ann, token, "Ann")).toEqual({ refused: "own" });
    expect(await joinExamination(ben, token, "")).toEqual({ refused: "name" });
    expect(await joinExamination(ben, token, "Ben")).toEqual({ id });
    expect(await joinExamination(ben, token, "Ben again")).toEqual({ id });
    expect(await joinExamination(ben, "no-such-token-xxxxxxxx", "Ben")).toEqual({ refused: "missing" });
    expect(await attest(cal, id, true, "Not an examiner.")).toBe(false);
    expect(await attest(ben, id, true, "She answered every objection from the deprivation account.")).toBe(true);
    expect(await attest(ben, id, false, "Changing my mind.")).toBe(false);
    const mine = await getExamination(ann, id);
    expect(mine).toMatchObject({ name: "Ann R.", courseTitle: "Death", status: "open", token });
    expect(mine!.examiners).toEqual([expect.objectContaining({ userId: ben, name: "Ben", attests: true, attestation: "She answered every objection from the deprivation account." })]);
    expect(await getExamination(ben, id)).toBeNull();
    expect((await examinationByToken(token))!.claim).toBe("Why death is bad for the one who dies.");
    expect((await listExaminations(ann, "https://oyc.yale.edu/philosophy/phil-176")).map((e) => e.id)).toEqual([id]);
    await closeExamination(ben, id);
    expect((await getExamination(ann, id))!.status).toBe("open");
    await closeExamination(ann, id);
    expect(await joinExamination(cal, token, "Cal")).toEqual({ refused: "closed" });
    expect((await getExamination(ann, id))!.status).toBe("closed");
  });
});
