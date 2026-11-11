import { randomUUID } from "node:crypto";
import type { CredentialInput } from "@reflect/core/credentials";
import type { LinkedInExport } from "@reflect/core/linkedin";
import type { CourseRow, TranscriptDraft } from "@reflect/core/transcript";
import { db } from "@/db/pool";
import { open, seal, tag } from "./crypto";
import { subjectId, subjects, tx, type SubjectOut } from "./learn-store";

/**
 * Sources of what a person knows beside books (Torian, 2026-11-11): a transcript, whose rows a model
 * drafts and the person confirms; credentials; and vouches from a LinkedIn export, which the person sees
 * before keeping. None of them set the person's grasp.
 */

export type Upload = { id: string; kind: "transcript" | "linkedin"; fileName: string; createdAt: string; transcript?: TranscriptDraft; linkedin?: LinkedInExport; kept: number[] };
export type CourseOut = { id: string; row: CourseRow & { institution?: string }; subject: SubjectOut };
export type CredentialOut = { id: string; name: string; issuer: string; issued?: string; expires?: string; url?: string; checked?: { name: string; issuer: string; issued?: string; at: string }; subjects: string[] };

const MAX_FILE = 5_000_000;

export async function saveUpload(userId: string, kind: Upload["kind"], fileName: string, body: TranscriptDraft | LinkedInExport, file?: Uint8Array): Promise<string> {
  if (file && file.byteLength > MAX_FILE) throw new Error("the file is larger than 5 MB");
  const id = randomUUID();
  await db().query("INSERT INTO learn_upload (id, user_id, kind, file_name_ct, file_ct, draft_ct) VALUES ($1, $2, $3, $4, $5, $6)", [id, userId, kind, seal(userId, fileName.slice(0, 200)), file ? seal(userId, Buffer.from(file).toString("base64")) : null, seal(userId, JSON.stringify(body))]);
  return id;
}

export async function getUpload(userId: string, id: string): Promise<Upload | null> {
  if (!/^[0-9a-f-]{36}$/.test(id)) return null;
  const r = await db().query("SELECT * FROM learn_upload WHERE id = $1 AND user_id = $2", [id, userId]);
  const u = r.rows[0];
  if (!u) return null;
  const kept = (await db().query("SELECT row_index FROM learn_course WHERE upload_id = $1", [id])).rows.map((x) => Number(x.row_index));
  const body = JSON.parse(open(userId, u.draft_ct));
  return { id, kind: u.kind, fileName: open(userId, u.file_name_ct), createdAt: new Date(u.created_at).toISOString(), ...(u.kind === "transcript" ? { transcript: body as TranscriptDraft } : { linkedin: body as LinkedInExport }), kept };
}

/** Keep the rows of a transcript draft the person confirmed, once each. */
export async function keepCourses(userId: string, uploadId: string, rows: number[]): Promise<number> {
  const u = await getUpload(userId, uploadId);
  if (!u?.transcript) throw new Error("no such transcript");
  let n = 0;
  await tx(async (c) => {
    for (const i of rows) {
      const row = u.transcript!.courses[i];
      if (!row || u.kept.includes(i)) continue;
      const sid = await subjectId(c, userId, row.title);
      await c.query("INSERT INTO learn_course (id, user_id, upload_id, row_index, subject_id, row_ct) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING", [randomUUID(), userId, uploadId, i, sid, seal(userId, JSON.stringify({ ...row, ...(u.transcript!.institution ? { institution: u.transcript!.institution } : {}) }))]);
      n++;
    }
  });
  return n;
}

/** Courses the person's AI read from their transcript and the person confirmed. */
export async function addCourses(userId: string, rows: (CourseRow & { institution?: string })[]): Promise<number> {
  await tx(async (c) => {
    for (const row of rows) {
      if (!row.title?.trim()) throw new Error("a course needs a title");
      await c.query("INSERT INTO learn_course (id, user_id, subject_id, row_ct) VALUES ($1, $2, $3, $4)", [randomUUID(), userId, await subjectId(c, userId, row.title), seal(userId, JSON.stringify(row))]);
    }
  });
  return rows.length;
}

export async function listCourses(userId: string): Promise<CourseOut[]> {
  const subs = await subjects(userId);
  const r = await db().query("SELECT id, subject_id, row_ct FROM learn_course WHERE user_id = $1 ORDER BY created_at, row_index", [userId]);
  return r.rows.map((x) => ({ id: x.id as string, row: JSON.parse(open(userId, x.row_ct as string)) as CourseOut["row"], subject: subs.get(x.subject_id as string)! }));
}

export async function addCredential(userId: string, name: string, c: CredentialInput): Promise<string | null> {
  if (!name.trim() || !c.issuer?.trim()) throw new Error("a credential has a name and an issuer");
  for (const d of [c.issued, c.expires]) if (d && !/^\d{4}(-\d{2}(-\d{2})?)?$/.test(d)) throw new Error("dates are YYYY, YYYY-MM or YYYY-MM-DD");
  if (c.url && !/^https?:\/\/\S+$/.test(c.url)) throw new Error("the link must start with http");
  const id = randomUUID();
  let kept = false;
  await tx(async (k) => {
    const body = { name: name.trim(), issuer: c.issuer.trim(), ...(c.issued ? { issued: c.issued } : {}), ...(c.expires ? { expires: c.expires } : {}), ...(c.url ? { url: c.url } : {}) };
    const r = await k.query("INSERT INTO learn_credential (id, user_id, dedupe_tag, body_ct, checked_ct) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (user_id, dedupe_tag) DO NOTHING", [id, userId, tag(userId, `credential|${body.issuer.toLowerCase()}|${body.name.toLowerCase()}`), seal(userId, JSON.stringify(body)), c.checked ? seal(userId, JSON.stringify(c.checked)) : null]);
    kept = (r.rowCount ?? 0) > 0;
    if (kept) for (const s of c.subjects ?? []) if (s.trim()) await k.query("INSERT INTO learn_credential_subject (credential_id, subject_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [id, await subjectId(k, userId, s)]);
  });
  return kept ? id : null;
}

export async function listCredentials(userId: string): Promise<CredentialOut[]> {
  const subs = await subjects(userId);
  const r = await db().query("SELECT id, body_ct, checked_ct FROM learn_credential WHERE user_id = $1 ORDER BY created_at", [userId]);
  const out: CredentialOut[] = [];
  for (const x of r.rows) {
    const links = await db().query("SELECT subject_id FROM learn_credential_subject WHERE credential_id = $1", [x.id]);
    out.push({ id: x.id, ...(JSON.parse(open(userId, x.body_ct)) as Omit<CredentialOut, "id" | "subjects">), ...(x.checked_ct ? { checked: JSON.parse(open(userId, x.checked_ct)) } : {}), subjects: links.rows.map((l) => subs.get(l.subject_id as string)?.name ?? "").filter(Boolean) });
  }
  return out;
}

export type LinkedInKinds = { endorsements?: boolean; recommendations?: boolean; certifications?: boolean; skills?: boolean };

/** Keep what of a LinkedIn export the person chose, once each. Returns how many new of each. */
export async function keepLinkedIn(userId: string, x: LinkedInExport, kinds: LinkedInKinds) {
  const counts = { endorsements: 0, recommendations: 0, certifications: 0, skills: 0 };
  await tx(async (c) => {
    if (kinds.endorsements) for (const e of x.endorsements) {
      const r = await c.query("INSERT INTO learn_vouch (id, user_id, kind, subject_id, source, dedupe_tag, body_ct) VALUES ($1, $2, 'endorsement', $3, 'linkedin', $4, $5) ON CONFLICT (user_id, dedupe_tag) DO NOTHING", [randomUUID(), userId, await subjectId(c, userId, e.skill), tag(userId, `endorsement|${e.skill.toLowerCase()}|${e.by}|${e.on}`), seal(userId, JSON.stringify(e))]);
      counts.endorsements += r.rowCount ?? 0;
    }
    if (kinds.recommendations) for (const rec of x.recommendations) {
      const r = await c.query("INSERT INTO learn_vouch (id, user_id, kind, source, dedupe_tag, body_ct) VALUES ($1, $2, 'recommendation', 'linkedin', $3, $4) ON CONFLICT (user_id, dedupe_tag) DO NOTHING", [randomUUID(), userId, tag(userId, `recommendation|${rec.by}|${rec.on ?? ""}`), seal(userId, JSON.stringify(rec))]);
      counts.recommendations += r.rowCount ?? 0;
    }
    if (kinds.skills) for (const s of x.skills) {
      await subjectId(c, userId, s);
      counts.skills++;
    }
  });
  if (kinds.certifications) for (const cert of x.certifications) if (await addCredential(userId, cert.name, { issuer: cert.issuer, ...(cert.issued ? { issued: cert.issued } : {}), ...(cert.expires ? { expires: cert.expires } : {}), ...(cert.url ? { url: cert.url } : {}) })) counts.certifications++;
  return counts;
}

export async function recommendations(userId: string): Promise<{ by: string; role?: string; company?: string; on?: string; text: string }[]> {
  const r = await db().query("SELECT body_ct FROM learn_vouch WHERE user_id = $1 AND kind = 'recommendation' ORDER BY created_at", [userId]);
  return r.rows.map((x) => JSON.parse(open(userId, x.body_ct as string)));
}
