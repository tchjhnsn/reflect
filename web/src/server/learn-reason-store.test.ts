import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { keepReason, reasonFor } from "./learn-reason-store";
import { getWork } from "./work-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

describe.skipIf(!live)("why a person is learning a course, on Neon", () => {
  afterAll(() => db().end());

  it("keeps the reason as a journal work, with origins and the kind of end, and a new version when it changes", async () => {
    const id = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, 'Ann', $2, false, now(), now())`, [id, `${id}@example.test`]);
    const url = "https://oyc.yale.edu/philosophy/phil-176";
    expect(await reasonFor(id, url)).toBeNull();
    await expect(keepReason(id, { courseUrl: url, courseTitle: "Death", cause: " ", end: "", origins: [], endKind: null })).rejects.toThrow();
    const workId = await keepReason(id, { courseUrl: url, courseTitle: "Death", cause: "My father died this year.", end: "To think it through.", origins: ["chance", "anger", "reasoning"], endKind: "own-sake" });
    expect(await reasonFor(id, url)).toMatchObject({ workId, cause: "My father died this year.", end: "To think it through.", origins: ["chance", "reasoning", "spirit"], endKind: "own-sake" });
    expect(await getWork(id, workId)).toMatchObject({ title: "Why I am learning Death", template: "why-learn", domain: "learn" });
    expect(await keepReason(id, { courseUrl: `${url}/`, courseTitle: "Death", cause: "My father died this year.", end: "To answer my brother.", origins: ["spirit"], endKind: "use" })).toBe(workId);
    expect(await reasonFor(id, url)).toMatchObject({ end: "To answer my brother.", origins: ["spirit"], endKind: "use" });
    expect((await getWork(id, workId))!.versions).toBe(3);
  });
});
