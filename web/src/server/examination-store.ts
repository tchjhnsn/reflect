// Examinations (migration 0026): a person puts themselves under examination on a course; people they invite examine
// them and attest in their own words, once. Everything written is sealed with the candidate's key, since the record is
// theirs; examiners read it through the server, never the database.
import { randomBytes, randomUUID } from "node:crypto";
import { EXAM, cleanText, joinRefusal, type ExamStatus, type JoinRefusal } from "@reflect/core/examination";
import { courseSite } from "@reflect/core/courses";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";
import { tx } from "./learn-store";

export type Examiner = { userId: string; name: string; joinedAt: string; attests: boolean | null; attestation: string | null; attestedAt: string | null };
export type Examination = {
  id: string; candidate: string; courseUrl: string; courseTitle: string; name: string; claim: string;
  status: ExamStatus; createdAt: string; closedAt: string | null; examiners: Examiner[];
};

type Row = { id: string; user_id: string; course_url: string; course_title: string; name_ct: string; claim_ct: string; status: ExamStatus; created_at: Date; closed_at: Date | null };
type ERow = { examination_id: string; user_id: string; name_ct: string; joined_at: Date; attests: boolean | null; attestation_ct: string | null; attested_at: Date | null };

const iso = (d: Date | null) => (d ? d.toISOString() : null);
const shape = (r: Row, ex: ERow[]): Examination => ({
  id: r.id, candidate: r.user_id, courseUrl: r.course_url, courseTitle: r.course_title,
  name: open(r.user_id, r.name_ct), claim: open(r.user_id, r.claim_ct), status: r.status, createdAt: r.created_at.toISOString(), closedAt: iso(r.closed_at),
  examiners: ex.filter((e) => e.examination_id === r.id).map((e) => ({
    userId: e.user_id, name: open(r.user_id, e.name_ct), joinedAt: e.joined_at.toISOString(), attests: e.attests,
    attestation: e.attestation_ct ? open(r.user_id, e.attestation_ct) : null, attestedAt: iso(e.attested_at),
  })),
});

async function load(where: string, args: unknown[]): Promise<Examination[]> {
  const rows = (await db().query<Row>(`SELECT id, user_id, course_url, course_title, name_ct, claim_ct, status, created_at, closed_at FROM examination WHERE ${where} ORDER BY created_at DESC`, args)).rows;
  if (!rows.length) return [];
  const ex = (await db().query<ERow>("SELECT examination_id, user_id, name_ct, joined_at, attests, attestation_ct, attested_at FROM examination_examiner WHERE examination_id = ANY($1) ORDER BY joined_at", [rows.map((r) => r.id)])).rows;
  return rows.map((r) => shape(r, ex));
}

/** Puts the person under examination on a course: what they will answer for, and the name examiners will see. */
export async function createExamination(userId: string, f: { courseUrl: string; courseTitle: string; name: unknown; claim: unknown }): Promise<{ id: string; token: string }> {
  const site = courseSite(f.courseUrl);
  const name = cleanText(f.name, EXAM.name);
  const claim = cleanText(f.claim, EXAM.claim);
  const title = cleanText(f.courseTitle, 300);
  if (!site || !name || !claim || !title) throw new Error("A course, your name and what you will answer for are needed.");
  const id = randomUUID();
  const token = randomBytes(18).toString("base64url");
  await db().query("INSERT INTO examination (id, user_id, course_url, course_title, name_ct, claim_ct, invite_token) VALUES ($1, $2, $3, $4, $5, $6, $7)", [id, userId, site.url, title, seal(userId, name), seal(userId, claim), token]);
  return { id, token };
}

/** The person's own examinations, all or on one course. */
export async function listExaminations(userId: string, courseUrl?: string): Promise<Examination[]> {
  const site = courseUrl ? courseSite(courseUrl) : null;
  return site ? load("user_id = $1 AND course_url = $2", [userId, site.url]) : load("user_id = $1", [userId]);
}

/** One examination as its candidate sees it, with its invitation. */
export async function getExamination(userId: string, id: string): Promise<(Examination & { token: string }) | null> {
  const [e] = await load("id = $1 AND user_id = $2", [id, userId]).catch(() => []);
  if (!e) return null;
  const token = (await db().query<{ invite_token: string }>("SELECT invite_token FROM examination WHERE id = $1", [id])).rows[0]!.invite_token;
  return { ...e, token };
}

/** An examination reached by its invitation, for someone asked to examine. */
export async function examinationByToken(token: string): Promise<Examination | null> {
  if (!/^[\w-]{16,64}$/.test(token)) return null;
  const [e] = await load("invite_token = $1", [token]);
  return e ?? null;
}

/** Joins as an examiner by the invitation. Refused for the candidate, a closed examination or a full one. */
export async function joinExamination(userId: string, token: string, name: unknown): Promise<{ id: string } | { refused: JoinRefusal | "name" | "missing" }> {
  const e = await examinationByToken(token);
  if (!e) return { refused: "missing" };
  const already = e.examiners.some((x) => x.userId === userId);
  const refused = joinRefusal({ candidate: e.candidate, status: e.status, examiners: e.examiners.length }, userId, already);
  if (refused) return { refused };
  if (already) return { id: e.id };
  const n = cleanText(name, EXAM.name);
  if (!n) return { refused: "name" };
  await db().query("INSERT INTO examination_examiner (examination_id, user_id, name_ct) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING", [e.id, userId, seal(e.candidate, n)]);
  return { id: e.id };
}

/** An examiner's attestation, in their own words: yes, they know it, or not yet. Written once and never edited. */
export async function attest(userId: string, id: string, attests: boolean, words: unknown): Promise<boolean> {
  const text = cleanText(words, EXAM.attestation);
  if (!text) return false;
  return tx(async (c) => {
    const e = (await c.query<{ user_id: string; status: ExamStatus }>("SELECT user_id, status FROM examination WHERE id = $1 FOR UPDATE", [id])).rows[0];
    if (!e || e.status !== "open") return false;
    const r = await c.query("UPDATE examination_examiner SET attests = $3, attestation_ct = $4, attested_at = now() WHERE examination_id = $1 AND user_id = $2 AND attested_at IS NULL", [id, userId, attests, seal(e.user_id, text)]);
    return (r.rowCount ?? 0) === 1;
  });
}

/** Closes an examination; the record stays as it stands, and nobody can join or attest after. */
export async function closeExamination(userId: string, id: string): Promise<void> {
  await db().query("UPDATE examination SET status = 'closed', closed_at = now() WHERE id = $1 AND user_id = $2 AND status = 'open'", [id, userId]);
}

/** An examination as one of its examiners sees it, or null for anyone who has not joined it. */
export async function examinationAsExaminer(userId: string, id: string): Promise<Examination | null> {
  const [e] = await load("id = $1", [id]).catch(() => []);
  return e && e.examiners.some((x) => x.userId === userId) ? e : null;
}
