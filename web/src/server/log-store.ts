// The log, read from the apps' own records (docs/design/REFLECT_UI_2026-11-12.md section 8). Nothing is copied
// or written here: each call opens Learn's books with their status history (migration 0013), claims with their stances, grasps, and items with their sessions and lessons (migrations 0023, 0024), and core turns them
// into acts. The apps behind the other tiles (migration 0029) give theirs too.
import { db } from "@/db/pool";
import { learnActs, type Act } from "@reflect/core/log";
import type { BookStatus, Grasp } from "@reflect/core/learning";
import type { Stance } from "@reflect/core/claims";
import { open } from "./crypto";
import { isMedium, mediumOf, type Medium } from "@reflect/core/ways";
import { lifeLogActs } from "@reflect/core/life";
import { lifeActs, lifeRecords, lifeStateChanges } from "./life-store";

const iso = (d: unknown) => new Date(d as string).toISOString();

/** Every act the person's records show, in no order (the page groups them by the person's own day). */
export async function logActs(userId: string): Promise<Act[]> {
  const [books, statuses, stances, grasps] = await Promise.all([
    db().query("SELECT id, title_ct, created_at, started_on::text AS started_on, finished_on::text AS finished_on FROM learn_book WHERE user_id = $1", [userId]),
    db().query("SELECT book_id, status, on_day::text AS on_day, created_at FROM learn_book_status WHERE user_id = $1", [userId]).catch(() => ({ rows: [] })),
    db().query("SELECT c.id, c.text_ct, s.stance, s.created_at FROM learn_claim c JOIN learn_claim_stance s ON s.claim_id = c.id WHERE c.user_id = $1", [userId]).catch(() => ({ rows: [] })),
    db().query("SELECT s.name_ct, g.grasp, g.created_at FROM learn_grasp g JOIN learn_subject s ON s.id = g.subject_id WHERE g.user_id = $1", [userId]),
  ]);
  const claims = new Map<string, { text: string; stances: { stance: Stance; at: string }[] }>();
  for (const r of stances.rows) {
    const c = claims.get(r.id as string) ?? { text: open(userId, r.text_ct as string), stances: [] };
    c.stances.push({ stance: r.stance as Stance, at: iso(r.created_at) });
    claims.set(r.id as string, c);
  }
  const history = new Map<string, { status: BookStatus; day: string; at: string }[]>();
  for (const r of statuses.rows) history.set(r.book_id as string, [...(history.get(r.book_id as string) ?? []), { status: r.status as BookStatus, day: r.on_day as string, at: iso(r.created_at) }]);
  const items = await db().query(
    `SELECT i.id, i.medium, i.body_ct, i.created_at, i.started_on::text AS started_on, i.finished_on::text AS finished_on,
       COALESCE((SELECT array_agg(s.day::text) FROM learn_item_session s WHERE s.item_id = i.id), '{}') AS sessions,
       COALESCE((SELECT array_agg(l.created_at) FROM learn_item_lesson l WHERE l.item_id = i.id), '{}') AS lessons
     FROM learn_item i WHERE i.user_id = $1 AND i.removed_at IS NULL`, [userId]).catch(() => ({ rows: [] as Record<string, unknown>[] }));
  const [records, states, acts] = await Promise.all([lifeRecords(userId), lifeStateChanges(userId), lifeActs(userId)]).catch(() => [[], [], []] as const);
  return [...lifeLogActs(records, states, acts), ...learnActs({
    items: items.rows.filter((i) => isMedium(i.medium)).map((i) => ({
      title: (JSON.parse(open(userId, i.body_ct as string)) as { title: string }).title, done: mediumOf(i.medium as Medium).done, addedAt: iso(i.created_at),
      startedOn: (i.started_on as string | null) ?? null, finishedOn: (i.finished_on as string | null) ?? null,
      sessions: (i.sessions as string[]) ?? [], lessons: ((i.lessons as unknown[]) ?? []).map(iso),
    })),
    books: books.rows.map((b) => ({ title: open(userId, b.title_ct as string), addedAt: iso(b.created_at), startedOn: (b.started_on as string | null) ?? null, finishedOn: (b.finished_on as string | null) ?? null, statuses: history.get(b.id as string) })),
    claims: [...claims.values()],
    grasps: grasps.rows.map((g) => ({ subject: open(userId, g.name_ct as string), grasp: g.grasp as Grasp, at: iso(g.created_at) })),
  })];
}
