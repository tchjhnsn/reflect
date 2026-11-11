import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { anchorCourses } from "@reflect/core/transcript";
import { readLinkedInExport } from "@reflect/core/linkedin";
import { db } from "@/db/pool";
import { learnView, rateSubject } from "./learn-store";
import { addCredential, getUpload, keepCourses, keepLinkedIn, listCourses, listCredentials, recommendations, saveUpload } from "./learn-sources-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

describe.skipIf(!live)("transcripts, credentials and vouches on Neon", () => {
  afterAll(() => db().end());

  it("keeps only confirmed transcript rows, credentials once, and LinkedIn vouches beside the person's grasp", async () => {
    const user = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [user, `${user}@example.test`]);
    const text = "STATE UNIVERSITY Fall 2016 MATH 221 Linear Algebra 3.0 B- PHIL 101 Intro to Ethics 3.0 A";
    const draft = anchorCourses(text, { institution: "State University", courses: [{ term: "Fall 2016", code: "MATH 221", title: "Linear Algebra", credits: 3, grade: "B-" }, { term: "Fall 2016", code: "PHIL 101", title: "Intro to Ethics", credits: 3, grade: "A" }, { term: "Fall 2016", title: "Made Up", grade: "A" }] });
    expect(draft.dropped).toEqual(["Made Up"]);
    const up = await saveUpload(user, "transcript", "t.pdf", draft, new Uint8Array([1, 2, 3]));
    expect(await keepCourses(user, up, [0])).toBe(1);
    expect(await keepCourses(user, up, [0])).toBe(0);
    expect((await getUpload(user, up))!.kept).toEqual([0]);
    const courses = await listCourses(user);
    expect(courses.map((c) => [c.row.title, c.row.grade, c.row.institution, c.subject.grasp])).toEqual([["Linear Algebra", "B-", "State University", null]]);
    await rateSubject(user, "Linear Algebra", "explain");
    expect((await listCourses(user))[0]!.subject.grasp?.grasp).toBe("explain");

    expect(await addCredential(user, "PMP", { issuer: "PMI", issued: "2021-03", subjects: ["Project management"] })).toBeTruthy();
    expect(await addCredential(user, "pmp", { issuer: "pmi" })).toBeNull();
    expect((await listCredentials(user)).map((c) => [c.name, c.issuer, c.subjects])).toEqual([["PMP", "PMI", ["Project management"]]]);

    const x = readLinkedInExport(new Map([
      ["Endorsement_Received_Info.csv", "Endorsement Date,Skill Name,Endorser First Name,Endorser Last Name,Endorser Public Url,Endorsement Status\n2024/03/01 10:00:00 UTC,Systems Thinking,Ana,Lee,,ACCEPTED\n"],
      ["Recommendations_Received.csv", 'First Name,Last Name,Company,Job Title,Text,Creation Date,Status\nAna,Lee,Acme,CTO,"Sees the whole system.",03/05/24,VISIBLE\n'],
    ]));
    expect(await keepLinkedIn(user, x, { endorsements: true, recommendations: true })).toEqual({ endorsements: 1, recommendations: 1, certifications: 0, skills: 0 });
    expect(await keepLinkedIn(user, x, { endorsements: true, recommendations: true })).toEqual({ endorsements: 0, recommendations: 0, certifications: 0, skills: 0 });
    expect((await recommendations(user))[0]!.text).toBe("Sees the whole system.");
    const v = await learnView(user);
    expect(v.known.map((s) => s.name)).toEqual(["Linear Algebra"]);
    const all = [...v.known];
    expect(all.some((s) => s.name === "Systems Thinking")).toBe(false);
  });
});
