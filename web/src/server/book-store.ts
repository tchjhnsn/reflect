// The book of your life (migration 0016): beliefs are Learn claims, placed in a domain by the person and,
// when marked in the journal editor, tied to the passage they came from. Migration 0017 marks each belief an
// ideal or a practical belief (a practical one names the ideal it serves and its circumstances) and keeps its
// origins. The book itself is computed by the core (@reflect/core/book) each time; an edition is a work.
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { bookMarkdown, checkKind, compileBook, isDomain, type Belief, type BeliefKind, type Domain } from "@reflect/core/book";
import type { Holder } from "@reflect/core/claims";
import { cleanOrigins, type Origin } from "@reflect/core/origins";
import { workTitle } from "@reflect/core/works";
import { open, seal } from "./crypto";
import { keepClaim, listClaims } from "./claims-store";
import { getWork, listWorks } from "./work-store";
import { subjectDomains } from "./subject-domain-store";

const isId = (id: string) => /^[0-9a-f-]{36}$/i.test(id);

/** Place one of the person's claims in a chapter. */
export async function placeClaim(userId: string, claimId: string, domain: string): Promise<void> {
  if (!isDomain(domain)) throw new Error("That is not a domain of the book.");
  if (!(await own(userId, claimId))) throw new Error("No such belief.");
  await db().query("INSERT INTO claim_domain (id, user_id, claim_id, domain) VALUES ($1, $2, $3, $4)", [randomUUID(), userId, claimId, domain]);
}

const own = async (userId: string, claimId: string) =>
  isId(claimId) && Boolean((await db().query("SELECT 1 FROM learn_claim WHERE id = $1 AND user_id = $2", [claimId, userId])).rows[0]);

/** What the person says of a belief when marking it. `serves` is an ideal they keep, or the words of a new one. */
export type Marking = {
  kind: BeliefKind | string;
  serves?: { id: string } | { text: string } | null;
  circumstances?: string | null;
  origins?: readonly string[];
};

/**
 * Mark one of the person's beliefs an ideal or a practical belief, with its origins. A practical belief
 * names the ideal it serves: one they already keep, or new words that become an ideal of its own, placed
 * in the same domain.
 */
export async function markBelief(userId: string, claimId: string, m: Marking): Promise<void> {
  if (!(await own(userId, claimId))) throw new Error("No such belief.");
  let serves: string | null = null;
  if (m.kind === "practical" && m.serves) {
    if ("id" in m.serves) {
      const ideals = await db().query("SELECT DISTINCT ON (claim_id) kind FROM claim_kind WHERE user_id = $1 AND claim_id = $2 ORDER BY claim_id, created_at DESC, id", [userId, m.serves.id]);
      if (!(await own(userId, m.serves.id)) || ideals.rows[0]?.kind !== "ideal") throw new Error("That is not one of your ideals.");
      serves = m.serves.id;
    } else if (m.serves.text.trim()) {
      checkKind(m.kind, "new", m.circumstances); // refuse before keeping a new ideal
      serves = await keepClaim(userId, { text: m.serves.text, stance: "belief" });
      if (serves === claimId) throw new Error("A practical belief cannot serve itself.");
      await db().query("INSERT INTO claim_kind (id, user_id, claim_id, kind) VALUES ($1, $2, $3, 'ideal')", [randomUUID(), userId, serves]);
      const d = await domainOf(userId, claimId);
      if (d) await placeClaim(userId, serves, d);
    }
  }
  if (serves === claimId) throw new Error("A practical belief cannot serve itself.");
  const { kind, circumstances } = checkKind(m.kind, serves, m.circumstances);
  await db().query("INSERT INTO claim_kind (id, user_id, claim_id, kind, serves_claim_id, circumstances_ct) VALUES ($1, $2, $3, $4, $5, $6)", [randomUUID(), userId, claimId, kind, kind === "practical" ? serves : null, circumstances ? seal(userId, circumstances) : null]);
  if (m.origins) await db().query("INSERT INTO claim_origin (id, user_id, claim_id, origins) VALUES ($1, $2, $3, $4)", [randomUUID(), userId, claimId, cleanOrigins(m.origins)]);
}

async function domainOf(userId: string, claimId: string): Promise<Domain | null> {
  const r = await db().query("SELECT domain FROM claim_domain WHERE user_id = $1 AND claim_id = $2 ORDER BY created_at DESC, id LIMIT 1", [userId, claimId]);
  return (r.rows[0]?.domain as Domain | undefined) ?? null;
}

/**
 * A passage the person selected in one of their works becomes a belief, marked an ideal or a practical
 * belief, placed in the domain they chose, with its origins and who else holds it, and tied back to the work.
 * The passage is the claim's words, as they wrote it.
 */
export async function beliefFromWork(userId: string, workId: string, passage: string, domain: string, m: Marking & { holders?: Holder[] }): Promise<string> {
  if (!isDomain(domain)) throw new Error("Choose a domain for it.");
  const work = await getWork(userId, workId);
  if (!work) throw new Error("No such work.");
  const text = passage.replace(/\s+/g, " ").trim();
  if (!text) throw new Error("Select the passage first.");
  checkKind(m.kind, m.serves && ("id" in m.serves ? m.serves.id : m.serves.text.trim()) ? "given" : null, m.circumstances);
  const claimId = await keepClaim(userId, { text, stance: "belief", holders: m.holders });
  await db().query("INSERT INTO claim_source (id, user_id, claim_id, work_id, passage_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, claimId, workId, seal(userId, text)]);
  await placeClaim(userId, claimId, domain);
  await markBelief(userId, claimId, m);
  return claimId;
}

/** Every claim as the book reads it: stance history, its domain (placed itself, else its Learn subject's), and the work it was marked from. */
export async function beliefs(userId: string): Promise<Belief[]> {
  const [claims, domains, sources, works, kinds, origins, bySubject] = await Promise.all([
    listClaims(userId),
    db().query("SELECT DISTINCT ON (claim_id) claim_id, domain FROM claim_domain WHERE user_id = $1 ORDER BY claim_id, created_at DESC, id", [userId]),
    db().query("SELECT DISTINCT ON (claim_id) claim_id, work_id FROM claim_source WHERE user_id = $1 ORDER BY claim_id, created_at", [userId]),
    listWorks(userId),
    db().query("SELECT DISTINCT ON (claim_id) claim_id, kind, serves_claim_id, circumstances_ct FROM claim_kind WHERE user_id = $1 ORDER BY claim_id, created_at DESC, id", [userId]),
    db().query("SELECT DISTINCT ON (claim_id) claim_id, origins FROM claim_origin WHERE user_id = $1 ORDER BY claim_id, created_at DESC, id", [userId]),
    subjectDomains(userId),
  ]);
  const textOf = new Map(claims.map((c) => [c.id, c.text]));
  const kindOf = new Map(kinds.rows.map((r) => [r.claim_id as string, r]));
  const originsOf = new Map(origins.rows.map((r) => [r.claim_id as string, cleanOrigins(r.origins as string[]) as Origin[]]));
  const domainIn = new Map(domains.rows.map((r) => [r.claim_id as string, r.domain as Domain]));
  const titles = new Map(works.map((w) => [w.id, workTitle(w)]));
  const sourceOf = new Map(sources.rows.map((r) => [r.claim_id as string, r.work_id as string]));
  return claims.map((c) => {
    const w = sourceOf.get(c.id);
    const k = kindOf.get(c.id);
    const serves = k?.serves_claim_id ? { id: k.serves_claim_id as string, text: textOf.get(k.serves_claim_id as string) ?? "" } : null;
    return {
      id: c.id, text: c.text, holders: c.holders, history: c.history, domain: domainIn.get(c.id) ?? (c.subject ? bySubject.get(c.subject.id) : undefined) ?? null, source: w ? { workId: w, title: titles.get(w) ?? "a work" } : null,
      kind: (k?.kind as BeliefKind | undefined) ?? null, serves, circumstances: k?.circumstances_ct ? open(userId, k.circumstances_ct as string) : null, origins: originsOf.get(c.id) ?? [],
    };
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


/** The ideals the person holds now, for a practical belief to name the one it serves. */
export async function idealsOf(userId: string): Promise<{ id: string; text: string; domain: Domain | null }[]> {
  const { chapters, unplaced } = await bookOf(userId);
  const held = [...chapters.flatMap((c) => c.entries.map((e) => ({ e, domain: c.domain as Domain | null }))), ...unplaced.map((e) => ({ e, domain: null }))];
  return held.filter(({ e }) => e.state === "held" && e.kind === "ideal").map(({ e, domain }) => ({ id: e.id, text: e.text, domain }));
}

/** One domain of a life: its works, and its chapter of the book (ideals, then practical beliefs). */
export async function domainView(userId: string, domain: Domain) {
  const [book, works] = await Promise.all([bookOf(userId), listWorks(userId)]);
  return { chapter: book.chapters.find((c) => c.domain === domain)!, works: works.filter((w) => w.domain === domain) };
}

/**
 * Send a passage the person wrote to Learn: it becomes a claim with their stance, filed under a Learn
 * subject (so it reaches whatever domain the subject is placed in), and tied back to the work it came from.
 */
export async function claimFromWork(userId: string, workId: string, passage: string, subject: string, stance: string): Promise<string> {
  const work = await getWork(userId, workId);
  if (!work) throw new Error("No such work.");
  const text = passage.replace(/\s+/g, " ").trim();
  if (!text) throw new Error("Select the passage first.");
  if (!subject.trim()) throw new Error("Name the subject it belongs under.");
  const claimId = await keepClaim(userId, { text, stance, subject });
  await db().query("INSERT INTO claim_source (id, user_id, claim_id, work_id, passage_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, claimId, workId, seal(userId, text)]);
  return claimId;
}

/** The work each claim was first sent or marked from, by claim id. */
export async function claimWorks(userId: string): Promise<Map<string, { workId: string; title: string }>> {
  const [rows, works] = await Promise.all([
    db().query("SELECT DISTINCT ON (claim_id) claim_id, work_id FROM claim_source WHERE user_id = $1 ORDER BY claim_id, created_at", [userId]),
    listWorks(userId),
  ]);
  const titles = new Map(works.map((w) => [w.id, workTitle(w)]));
  return new Map(rows.rows.map((r) => [r.claim_id as string, { workId: r.work_id as string, title: titles.get(r.work_id as string) ?? "a work" }]));
}
