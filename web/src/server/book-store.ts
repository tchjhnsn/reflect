// The book of your life (migration 0016): beliefs are Learn claims, placed in a domain by the person and,
// when marked in the journal editor, tied to the passage they came from. The book itself is computed by the
// core (@reflect/core/book) each time; an edition is kept as a work.
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { bookMarkdown, compileBook, isDomain, type Belief, type Domain } from "@reflect/core/book";
import { workTitle } from "@reflect/core/works";
import { seal } from "./crypto";
import { keepClaim, listClaims } from "./claims-store";
import { getWork, listWorks } from "./work-store";

const isId = (id: string) => /^[0-9a-f-]{36}$/i.test(id);

/** Place one of the person's claims in a chapter. */
export async function placeClaim(userId: string, claimId: string, domain: string): Promise<void> {
  if (!isDomain(domain)) throw new Error("That is not a domain of the book.");
  if (!isId(claimId) || !(await db().query("SELECT 1 FROM learn_claim WHERE id = $1 AND user_id = $2", [claimId, userId])).rows[0]) throw new Error("No such belief.");
  await db().query("INSERT INTO claim_domain (id, user_id, claim_id, domain) VALUES ($1, $2, $3, $4)", [randomUUID(), userId, claimId, domain]);
}

/**
 * "I hold this": a passage the person selected in one of their works becomes a belief they hold, placed in
 * the domain they chose and tied back to the work. The passage is the claim's words, as they wrote it.
 */
export async function holdFromWork(userId: string, workId: string, passage: string, domain: string): Promise<string> {
  if (!isDomain(domain)) throw new Error("Choose a chapter for it.");
  const work = await getWork(userId, workId);
  if (!work) throw new Error("No such work.");
  const text = passage.replace(/\s+/g, " ").trim();
  if (!text) throw new Error("Select what you hold first.");
  const claimId = await keepClaim(userId, { text, stance: "belief" });
  await db().query("INSERT INTO claim_source (id, user_id, claim_id, work_id, passage_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, claimId, workId, seal(userId, text)]);
  await placeClaim(userId, claimId, domain);
  return claimId;
}

/** Every claim as the book reads it: stance history, latest domain, and the work it was marked from. */
export async function beliefs(userId: string): Promise<Belief[]> {
  const [claims, domains, sources, works] = await Promise.all([
    listClaims(userId),
    db().query("SELECT DISTINCT ON (claim_id) claim_id, domain FROM claim_domain WHERE user_id = $1 ORDER BY claim_id, created_at DESC, id", [userId]),
    db().query("SELECT DISTINCT ON (claim_id) claim_id, work_id FROM claim_source WHERE user_id = $1 ORDER BY claim_id, created_at", [userId]),
    listWorks(userId),
  ]);
  const domainOf = new Map(domains.rows.map((r) => [r.claim_id as string, r.domain as Domain]));
  const titles = new Map(works.map((w) => [w.id, workTitle(w)]));
  const sourceOf = new Map(sources.rows.map((r) => [r.claim_id as string, r.work_id as string]));
  return claims.map((c) => {
    const w = sourceOf.get(c.id);
    return { id: c.id, text: c.text, holders: c.holders, history: c.history, domain: domainOf.get(c.id) ?? null, source: w ? { workId: w, title: titles.get(w) ?? "a work" } : null };
  });
}

export async function bookOf(userId: string) {
  return compileBook(await beliefs(userId));
}

/** Compile an edition: the book as it stands, kept as a dated memoir in the journal. */
export async function compileEdition(userId: string): Promise<string> {
  const { chapters } = await bookOf(userId);
  const now = new Date().toISOString();
  const id = randomUUID();
  const c = await db().connect();
  try {
    await c.query("BEGIN");
    await c.query("INSERT INTO work (id, user_id, kind, template) VALUES ($1, $2, 'memoir', 'book-edition')", [id, userId]);
    await c.query("INSERT INTO work_version (work_id, seq, title_ct, body_ct) VALUES ($1, 0, $2, $3)", [id, seal(userId, `The book of my life, ${now.slice(0, 10)}`), seal(userId, bookMarkdown(chapters, now))]);
    await c.query("COMMIT");
    return id;
  } catch (e) {
    await c.query("ROLLBACK");
    throw e;
  } finally {
    c.release();
  }
}

