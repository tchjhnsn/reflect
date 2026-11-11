// Credentials and vouches: what others say about what a person knows. A credential (a certificate, a
// license, a badge) is kept in learn/credentials/ with its issuer and the subjects it covers; a badge link
// in the Open Badges format can be checked against what the issuer publishes, and checking is optional.
// A vouch is someone saying the person knows a subject, such as a LinkedIn endorsement: a dated line on
// the subject. A recommendation is someone's words about the person, kept whole in learn/vouches/.
// None of these set the person's grasp; they stand beside it.
import { join } from "node:path";
import { keepSubject, slugOf } from "./learning.ts";
import { headOf } from "./lifetime.ts";
import type { Fetch } from "./openlibrary.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const CREDENTIALS = join("learn", "credentials");
export const VOUCHES = join("learn", "vouches");

export type Credential = { id: string; name: string; issuer: string; issued?: string; expires?: string; url?: string; subjects: string[]; checked?: { name: string; issuer: string; issued?: string; at: string } };
export type Vouch = { on: string; by: string; url?: string; source: string };
export type Recommendation = { by: string; role?: string; company?: string; on?: string; text: string; source: string };

const DAY = /^\d{4}(-\d{2}(-\d{2})?)?$/;
const q = (s: string) => JSON.stringify(s);
const oneLine = (s: string) => s.replace(/\s+/g, " ").trim();

/** What an Open Badges assertion at a link says, or null when the link is not one. */
export async function checkBadge(url: string, fetcher: Fetch): Promise<{ name: string; issuer: string; issued?: string } | null> {
  const r = await fetcher(url).catch(() => null);
  if (!r?.ok) return null;
  const a = (await r.json().catch(() => null)) as Record<string, unknown> | null;
  if (!a || !JSON.stringify(a["@context"] ?? "").includes("openbadges")) return null;
  let badge = a.badge as Record<string, unknown> | string | undefined;
  if (typeof badge === "string") badge = ((await (await fetcher(badge).catch(() => null))?.json().catch(() => null)) ?? undefined) as Record<string, unknown> | undefined;
  const credential = (a.credentialSubject as { achievement?: Record<string, unknown> } | undefined)?.achievement;
  const b = (badge ?? credential) as Record<string, unknown> | undefined;
  const issuerRaw = (b?.issuer ?? a.issuer) as Record<string, unknown> | string | undefined;
  const issuer = typeof issuerRaw === "string" ? issuerRaw : String(issuerRaw?.name ?? "");
  const name = String(b?.name ?? "");
  if (!name) return null;
  const issued = String(a.issuedOn ?? a.validFrom ?? a.issuanceDate ?? "").slice(0, 10);
  return { name, issuer, ...(DAY.test(issued) ? { issued } : {}) };
}

export type CredentialInput = { issuer: string; issued?: string; expires?: string; url?: string; subjects?: string[]; checked?: Credential["checked"] };

export async function keepCredential(root: string, name: string, c: CredentialInput, now: Date): Promise<Credential> {
  if (!name.trim() || !c.issuer?.trim()) throw new Error("a credential has a name and an issuer (--issuer)");
  for (const d of [c.issued, c.expires]) if (d && !DAY.test(d)) throw new Error("dates are YYYY, YYYY-MM or YYYY-MM-DD");
  if (c.url && !/^https?:\/\/\S+$/.test(c.url)) throw new Error("--url is a link");
  const id = slugOf(`${c.issuer} ${name}`);
  const rel = join(CREDENTIALS, `${id}.md`);
  const old = await readRecord(root, rel);
  const prev = old ? parseCredential(id, old) : null;
  const subjects = [...new Set([...(prev?.subjects ?? []), ...(await Promise.all((c.subjects ?? []).map(async (s) => (await keepSubject(root, s, {}, now)).id)))])];
  const cred: Credential = { id, name: name.trim(), issuer: c.issuer.trim(), ...(c.issued ? { issued: c.issued } : {}), ...(c.expires ? { expires: c.expires } : {}), ...(c.url ? { url: c.url } : {}), subjects, ...(c.checked ?? prev?.checked ? { checked: c.checked ?? prev!.checked } : {}) };
  const head = [`credential: ${id}`, `name: ${q(cred.name)}`, `issuer: ${q(cred.issuer)}`];
  if (cred.issued) head.push(`issued: ${cred.issued}`);
  if (cred.expires) head.push(`expires: ${cred.expires}`);
  if (cred.url) head.push(`url: ${cred.url}`);
  if (cred.subjects.length) head.push(`subjects: ${cred.subjects.map((s) => `[[learn/subjects/${s}]]`).join(", ")}`);
  if (cred.checked) head.push(`checked: ${q(JSON.stringify(cred.checked))}`);
  const at = now.toISOString();
  const line = cred.checked && cred.checked.at === c.checked?.at ? `${at.slice(0, 10)} checked: the issuer's badge says ${oneLine(cred.checked.name)} from ${oneLine(cred.checked.issuer)}` : `${at.slice(0, 10)} kept`;
  const history = old ? old.split("\n").filter((l) => l.startsWith("- ")) : [];
  await writeRecord(root, rel, `---\n${head.join("\n")}\n---\n\n# ${cred.name}\n\n${[...history, `- ${line}`].join("\n")}\n`, old ? "append" : "create", at);
  return cred;
}

export function parseCredential(id: string, text: string): Credential {
  const h = headOf(text);
  return {
    id,
    name: h.name ?? id,
    issuer: h.issuer ?? "",
    ...(h.issued ? { issued: h.issued } : {}),
    ...(h.expires ? { expires: h.expires } : {}),
    ...(h.url ? { url: h.url } : {}),
    subjects: [...(h.subjects ?? "").matchAll(/\[\[learn\/subjects\/([^\]]+)\]\]/g)].map((m) => m[1]!),
    ...(h.checked ? { checked: JSON.parse(h.checked) as Credential["checked"] } : {}),
  };
}

/** A vouch line on a subject: `2024-03-01 vouched by Ana Lee <https://…> on linkedin`. */
export function parseVouch(line: string): Vouch | null {
  const m = /^(\d{4}(?:-\d{2}(?:-\d{2})?)?) vouched by (.+?)(?: <([^>]+)>)? on ([a-z-]+)$/.exec(line);
  return m ? { on: m[1]!, by: m[2]!, ...(m[3] ? { url: m[3] } : {}), source: m[4]! } : null;
}

/** Add vouches to a subject, once each (same day, person and source). */
export async function keepVouches(root: string, subject: string, vouches: readonly Vouch[], now: Date): Promise<number> {
  const s = await keepSubject(root, subject, {}, now);
  const rel = join("learn", "subjects", `${s.id}.md`);
  const text = (await readRecord(root, rel))!;
  const have = new Set(text.split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2)));
  const add = vouches.map((v) => `${v.on} vouched by ${oneLine(v.by)}${v.url ? ` <${v.url}>` : ""} on ${v.source}`).filter((l) => !have.has(l));
  if (!add.length) return 0;
  const at = now.toISOString();
  await writeRecord(root, rel, `${text.replace(/\n*$/, "")}\n${add.map((l) => `- ${l}`).join("\n")}\n`, "append", at);
  return add.length;
}

/** Keep a recommendation whole, in the recommender's words. */
export async function keepRecommendation(root: string, r: Recommendation, now: Date): Promise<string> {
  if (!r.by.trim() || !r.text.trim()) throw new Error("a recommendation has who wrote it and their words");
  const rel = join(VOUCHES, `${slugOf(`${r.by} ${r.on ?? ""}`)}.md`);
  const old = await readRecord(root, rel);
  const head = [`by: ${q(r.by)}`, `source: ${r.source}`];
  if (r.role) head.push(`role: ${q(r.role)}`);
  if (r.company) head.push(`company: ${q(r.company)}`);
  if (r.on) head.push(`on: ${r.on}`);
  const at = now.toISOString();
  const bytes = `---\n${head.join("\n")}\n---\n\n# From ${r.by}\n\n${r.text.trim()}\n`;
  if (old === bytes) return rel;
  await writeRecord(root, rel, bytes, old ? "append" : "create", at);
  return rel;
}
