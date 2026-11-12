// The log, read from the apps' own records (docs/design/REFLECT_UI_2026-11-12.md section 8). Nothing is copied
// or written here: each call opens Learn's books, claims with their stances, and grasps, and core turns them
// into acts.
import { db } from "@/db/pool";
import { learnActs, type Act } from "@reflect/core/log";
import type { Grasp } from "@reflect/core/learning";
import type { Stance } from "@reflect/core/claims";
import { open } from "./crypto";

const iso = (d: unknown) => new Date(d as string).toISOString();

/** Every act the person's records show, in no order (the page groups them by the person's own day). */
export async function logActs(userId: string): Promise<Act[]> {
  const [books, stances, grasps] = await Promise.all([
    db().query("SELECT title_ct, created_at, started_on::text AS started_on, finished_on::text AS finished_on FROM learn_book WHERE user_id = $1", [userId]),
    db().query("SELECT c.id, c.text_ct, s.stance, s.created_at FROM learn_claim c JOIN learn_claim_stance s ON s.claim_id = c.id WHERE c.user_id = $1", [userId]).catch(() => ({ rows: [] })),
    db().query("SELECT s.name_ct, g.grasp, g.created_at FROM learn_grasp g JOIN learn_subject s ON s.id = g.subject_id WHERE g.user_id = $1", [userId]),
  ]);
  const claims = new Map<string, { text: string; stances: { stance: Stance; at: string }[] }>();
  for (const r of stances.rows) {
    const c = claims.get(r.id as string) ?? { text: open(userId, r.text_ct as string), stances: [] };
    c.stances.push({ stance: r.stance as Stance, at: iso(r.created_at) });
    claims.set(r.id as string, c);
  }
  return learnActs({
    books: books.rows.map((b) => ({ title: open(userId, b.title_ct as string), addedAt: iso(b.created_at), startedOn: (b.started_on as string | null) ?? null, finishedOn: (b.finished_on as string | null) ?? null })),
    claims: [...claims.values()],
    grasps: grasps.rows.map((g) => ({ subject: open(userId, g.name_ct as string), grasp: g.grasp as Grasp, at: iso(g.created_at) })),
  });
}
