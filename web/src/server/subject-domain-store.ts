// Learn subjects placed in the domains of a life (migration 0018). Learn stays the source: a domain shows the
// subjects placed in it, with their books and claims, read from Learn each time. Rows are only added.
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { isDomain, type Domain } from "@reflect/core/book";
import type { Stance } from "@reflect/core/claims";
import { listClaims } from "./claims-store";
import { learnView } from "./learn-store";
import { open } from "./crypto";

/** Every subject's domain now (the latest placement), by subject id; subjects never placed, or taken out, are absent. */
export async function subjectDomains(userId: string): Promise<Map<string, Domain>> {
  const r = await db().query("SELECT DISTINCT ON (subject_id) subject_id, domain FROM subject_domain WHERE user_id = $1 ORDER BY subject_id, created_at DESC, id", [userId]);
  return new Map(r.rows.filter((x) => isDomain(x.domain)).map((x) => [x.subject_id as string, x.domain as Domain]));
}

/** Place one of the person's subjects in a domain, or take it out of every domain (null). */
export async function placeSubject(userId: string, subjectId: string, domain: string | null): Promise<void> {
  if (domain !== null && !isDomain(domain)) throw new Error("That is not a domain.");
  if (!/^[0-9a-f-]{36}$/i.test(subjectId) || !(await db().query("SELECT 1 FROM learn_subject WHERE id = $1 AND user_id = $2", [subjectId, userId])).rows[0]) throw new Error("No such subject.");
  if (((await subjectDomains(userId)).get(subjectId) ?? null) === domain) return;
  await db().query("INSERT INTO subject_domain (id, user_id, subject_id, domain) VALUES ($1, $2, $3, $4)", [randomUUID(), userId, subjectId, domain]);
}

export type PlacedSubject = {
  id: string;
  name: string;
  books: { id: string; title: string; authors: string[]; status: string }[];
  claims: { id: string; text: string; stance: Stance | null }[];
};

/** A domain as Learn fills it: the subjects placed there with their books and claims, and the subjects not placed anywhere. */
export async function learnIn(userId: string, domain: Domain): Promise<{ subjects: PlacedSubject[]; unplaced: { id: string; name: string }[] }> {
  const [placed, view, claims, names] = await Promise.all([
    subjectDomains(userId),
    learnView(userId),
    listClaims(userId),
    db().query("SELECT id, name_ct FROM learn_subject WHERE user_id = $1", [userId]),
  ]);
  const nameOf = new Map(names.rows.map((r) => [r.id as string, open(userId, r.name_ct as string)]));
  const ids = [...nameOf.keys()];
  const byName = (a: { name: string }, b: { name: string }) => a.name.localeCompare(b.name);
  return {
    subjects: ids.filter((id) => placed.get(id) === domain).map((id) => ({
      id,
      name: nameOf.get(id)!,
      books: view.books.filter((b) => b.subjects.some((s) => s.id === id)).map((b) => ({ id: b.id, title: b.title, authors: b.authors, status: b.status })),
      claims: claims.filter((c) => c.subject?.id === id).map((c) => ({ id: c.id, text: c.text, stance: c.stance?.stance ?? null })),
    })).sort(byName),
    unplaced: ids.filter((id) => !placed.has(id)).map((id) => ({ id, name: nameOf.get(id)! })).sort(byName),
  };
}

/** The names of the person's Learn subjects, in order, for choosing one. */
export async function subjectNames(userId: string): Promise<string[]> {
  const r = await db().query("SELECT name_ct FROM learn_subject WHERE user_id = $1", [userId]);
  return r.rows.map((x) => open(userId, x.name_ct as string)).sort((a, b) => a.localeCompare(b));
}
