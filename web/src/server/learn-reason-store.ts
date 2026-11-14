// Why the person is learning a course (migration 0027). The writing is a work of their journal, so it is kept in
// versions and can be changed there; the origins they marked and the kind of end are a new row each time, and the
// latest counts.
import { randomUUID } from "node:crypto";
import { courseSite } from "@reflect/core/courses";
import { cleanOrigins, type Origin } from "@reflect/core/origins";
import { isEndKind, reasonBody, reasonParts, type EndKind } from "@reflect/core/learn-reason";
import { db } from "@/db/pool";
import { createWork, getWork, saveWork } from "./work-store";

export type Reason = { workId: string; cause: string; end: string; origins: Origin[]; endKind: EndKind | null; savedAt: string };

/** The person's latest reason for learning a course, or null before they have written one. */
export async function reasonFor(userId: string, courseUrl: string): Promise<Reason | null> {
  const site = courseSite(courseUrl);
  if (!site) return null;
  const r = (await db().query<{ work_id: string; origins: string[]; end_kind: string | null }>("SELECT work_id, origins, end_kind FROM learn_reason WHERE user_id = $1 AND course_url = $2 ORDER BY created_at DESC, id LIMIT 1", [userId, site.url])).rows[0];
  if (!r) return null;
  const w = await getWork(userId, r.work_id);
  if (!w) return null;
  return { workId: w.id, ...reasonParts(w.body), origins: cleanOrigins(r.origins), endKind: isEndKind(r.end_kind) ? r.end_kind : null, savedAt: w.savedAt };
}

/** Keeps what is causing the person to learn a course and for the sake of what: a journal work the first time, a new version after. */
export async function keepReason(userId: string, f: { courseUrl: string; courseTitle: string; cause: unknown; end: unknown; origins: readonly unknown[]; endKind: unknown }): Promise<string> {
  const site = courseSite(f.courseUrl);
  if (!site) throw new Error("That is not a course Reflect can read.");
  if (typeof f.cause !== "string" || !f.cause.trim()) throw new Error("Write what is causing you to learn this.");
  const prior = await reasonFor(userId, site.url);
  const workId = prior?.workId ?? await createWork(userId, "why-learn", null, "learn");
  await saveWork(userId, workId, `Why I am learning ${String(f.courseTitle).trim().slice(0, 200) || "this course"}`, reasonBody(f.cause, f.end));
  await db().query("INSERT INTO learn_reason (id, user_id, course_url, work_id, origins, end_kind) VALUES ($1, $2, $3, $4, $5, $6)", [randomUUID(), userId, site.url, workId, cleanOrigins(f.origins), isEndKind(f.endKind) ? f.endKind : null]);
  return workId;
}
