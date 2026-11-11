// Official documents as a custody checklist. Reflect does not need to hold a birth certificate or a
// Social Security card: the person checks off that they have custody of it, and where. For each kind
// of document the catalog says how it is normally kept today and what Reflect recommends: keep the
// paper safe, store it digitally, or connect the service that already holds it. A connection is a
// record that names a service (a secure vault, a school's credential service) and the document types
// it holds; Reflect records only that it holds them and never copies a document into the vault.
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { RecordPaths } from "./layout.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const DOCUMENT_GROUPS = ["identity", "education", "household", "money", "health", "legal", "legacy"] as const;
export type DocumentGroup = (typeof DOCUMENT_GROUPS)[number];

/** How a kind of document is normally kept today. */
export type Kept = "paper" | "digital" | "both";
export type Recommendation = "keep-paper-safe" | "store-digital" | "connect";

export type DocumentType = {
  id: string;
  name: string;
  group: DocumentGroup;
  kept: Kept;
  /** How it is normally kept, in a sentence. */
  norm: string;
  recommend: Recommendation;
  /** Kinds of service that hold it, for a connection. Examples, not endorsements. */
  services?: string[];
};

/**
 * The catalog, grouped after the categories of FEMA's Emergency Financial First Aid Kit (household
 * identification, financial and legal documents, medical information, insurance). The norms are as of
 * 2026-11 in the United States and change: a state that starts issuing a mobile license changes one.
 */
export const DOCUMENT_TYPES: readonly DocumentType[] = [
  { id: "birth-certificate", name: "Birth certificate", group: "identity", kept: "paper", recommend: "keep-paper-safe", norm: "A certified paper copy from the state or county vital records office; there is no official digital version." },
  { id: "social-security-card", name: "Social Security card", group: "identity", kept: "paper", recommend: "keep-paper-safe", norm: "Paper only. The Social Security Administration issues no digital card and advises not carrying it; your my Social Security account shows your record." },
  { id: "passport", name: "Passport", group: "identity", kept: "paper", recommend: "keep-paper-safe", norm: "A paper book (and an optional card) from the State Department." },
  { id: "drivers-license", name: "Driver's license or state ID", group: "identity", kept: "both", recommend: "store-digital", norm: "A physical card; a growing number of states also issue a mobile license in Apple Wallet, Google Wallet or a state app, which TSA accepts.", services: ["phone wallet", "state DMV app"] },
  { id: "military-discharge", name: "Military discharge (DD-214)", group: "identity", kept: "paper", recommend: "keep-paper-safe", norm: "Paper; the National Archives and the VA can provide copies." },
  { id: "diploma", name: "Diplomas", group: "education", kept: "both", recommend: "connect", norm: "A paper original, and many schools also issue a certified digital diploma through a credential service.", services: ["school credential service (Parchment)"] },
  { id: "transcripts", name: "Transcripts", group: "education", kept: "digital", recommend: "connect", norm: "Ordered and sent electronically through the school or a credential service.", services: ["Parchment", "National Student Clearinghouse"] },
  { id: "certifications", name: "Certifications and professional licenses", group: "education", kept: "digital", recommend: "connect", norm: "Often issued as digital badges or looked up in a licensing board's registry.", services: ["digital badge service (Credly)", "licensing board registry"] },
  { id: "marriage-certificate", name: "Marriage certificate", group: "household", kept: "paper", recommend: "keep-paper-safe", norm: "A certified paper copy from the county where the license was issued." },
  { id: "divorce-decree", name: "Divorce decree", group: "household", kept: "paper", recommend: "keep-paper-safe", norm: "A court record; certified copies come from the court clerk." },
  { id: "deed", name: "Property deed", group: "household", kept: "both", recommend: "keep-paper-safe", norm: "The original is paper; the county recorder keeps the recorded copy, often searchable online." },
  { id: "lease", name: "Lease", group: "household", kept: "digital", recommend: "store-digital", norm: "Usually signed electronically and kept as a PDF." },
  { id: "vehicle-title", name: "Vehicle titles", group: "household", kept: "paper", recommend: "keep-paper-safe", norm: "Paper in most states; while a loan is open some states hold it electronically with the lender." },
  { id: "tax-returns", name: "Tax returns", group: "money", kept: "digital", recommend: "store-digital", norm: "Filed and kept electronically; the IRS Online Account has transcripts. Keep at least three years, longer in some cases.", services: ["IRS Online Account", "tax software account"] },
  { id: "insurance-policies", name: "Insurance policies", group: "money", kept: "digital", recommend: "connect", norm: "Held in each insurer's portal or app.", services: ["insurer portal"] },
  { id: "account-statements", name: "Bank and investment statements", group: "money", kept: "digital", recommend: "connect", norm: "Held in each institution's online account.", services: ["bank or brokerage account"] },
  { id: "medical-records", name: "Medical records", group: "health", kept: "digital", recommend: "connect", norm: "Held in each provider's patient portal; phones can gather them in one place.", services: ["patient portal (MyChart)", "phone health records"] },
  { id: "immunization-record", name: "Immunization record", group: "health", kept: "digital", recommend: "connect", norm: "Kept in your state's immunization registry and your provider's portal.", services: ["state immunization registry", "patient portal"] },
  { id: "health-insurance-card", name: "Health insurance card", group: "health", kept: "both", recommend: "store-digital", norm: "A plastic card, with a digital card in the insurer's app.", services: ["insurer app"] },
  { id: "will", name: "Will", group: "legacy", kept: "paper", recommend: "keep-paper-safe", norm: "A signed paper original; a few states also allow electronic wills. Tell your executor where it is." },
  { id: "power-of-attorney", name: "Power of attorney", group: "legal", kept: "paper", recommend: "keep-paper-safe", norm: "Signed and usually notarized on paper; give copies to the people named." },
  { id: "advance-directive", name: "Advance directive", group: "legal", kept: "paper", recommend: "store-digital", norm: "Signed on paper; keep a digital copy where doctors and family can reach it, and some states keep a registry." },
  { id: "life-insurance", name: "Life insurance", group: "legacy", kept: "digital", recommend: "connect", norm: "Held in the insurer's portal or your employer's benefits site; the people named need to know it exists and who the insurer is.", services: ["insurer portal", "employer benefits site"] },
  { id: "beneficiary-designations", name: "Beneficiary designations", group: "legacy", kept: "digital", recommend: "connect", norm: "Set on each retirement account, insurance policy and bank account, and they pass outside the will, so they need checking whenever life changes.", services: ["each account's online settings"] },
  { id: "trust", name: "Trust", group: "legacy", kept: "paper", recommend: "keep-paper-safe", norm: "A signed trust document, usually kept by you and the attorney who drafted it." },
  { id: "funeral-plan", name: "Funeral plan", group: "legacy", kept: "paper", recommend: "keep-paper-safe", norm: "A prepaid contract with a funeral home if you have one; your wishes themselves go in legacy/wishes.md." },
  { id: "organ-donor-registration", name: "Organ donor registration", group: "legacy", kept: "digital", recommend: "store-digital", norm: "Recorded in your state's donor registry, often when you get a license, and shown on the license.", services: ["state donor registry"] },
  { id: "digital-legacy", name: "Digital legacy contacts", group: "legacy", kept: "digital", recommend: "store-digital", norm: "Set inside each service: Apple's Legacy Contact, Google's Inactive Account Manager, Facebook's legacy contact. Name who may reach your accounts; never write passwords here.", services: ["Apple Legacy Contact", "Google Inactive Account Manager"] },
];

export const documentType = (id: string): DocumentType | undefined => DOCUMENT_TYPES.find((d) => d.id === id);

/** What the person says about one document. */
export const CUSTODY = ["original", "digital-copy", "connection", "missing", "not-applicable"] as const;
export type Custody = (typeof CUSTODY)[number];
export const isCustody = (s: string): s is Custody => (CUSTODY as readonly string[]).includes(s);

export type CustodyCard = { type: string; custody: Custody; where?: string; connection?: string; checkedAt: string };

/** A custody card: front matter for the current state, and one line per check-off below, never removed. */
export function cardText(card: CustodyCard, history: readonly string[]): string {
  const head = [`type: ${card.type}`, `custody: ${card.custody}`];
  if (card.where) head.push(`where: ${JSON.stringify(card.where)}`);
  if (card.connection) head.push(`connection: ${card.connection}`);
  head.push(`checked_at: ${card.checkedAt}`);
  const name = documentType(card.type)?.name ?? card.type;
  return `---\n${head.join("\n")}\n---\n\n# ${name}\n\n${history.map((h) => `- ${h}`).join("\n")}\n`;
}

const historyLine = (c: CustodyCard): string =>
  `${c.checkedAt.slice(0, 10)} ${c.custody}${c.where ? `: ${c.where}` : ""}${c.connection ? ` (connection ${c.connection})` : ""}`;

export function parseCard(text: string): { card: Partial<CustodyCard>; history: string[] } {
  const body = text.replace(/\r\n/g, "\n");
  const card: Record<string, string> = {};
  let rest = body;
  if (body.startsWith("---\n")) {
    const end = body.indexOf("\n---\n", 4);
    if (end > 0) {
      for (const line of body.slice(4, end).split("\n")) {
        const m = /^([a-z_]+):\s*(.*)$/.exec(line);
        if (!m || !m[2]) continue;
        let v = m[2];
        if (v.startsWith('"')) {
          try {
            v = JSON.parse(v) as string;
          } catch {
            // keep as written
          }
        }
        card[m[1] === "checked_at" ? "checkedAt" : m[1]!] = v;
      }
      rest = body.slice(end + 5);
    }
  }
  const history = rest.split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2));
  return { card: card as Partial<CustodyCard>, history };
}

/** A service that holds records for the person, or a learning platform they use. Credentials never go here. */
export const CONNECTION_KINDS = ["vault", "issuer", "portal", "platform"] as const;
export type Connection = { id: string; name: string; kind: (typeof CONNECTION_KINDS)[number]; holds: string[]; created_at: string };

export function connectionProblems(c: Partial<Connection>, file: string): string[] {
  const out: string[] = [];
  if (typeof c.id !== "string" || `${c.id}.json` !== file) out.push(`connections/${file}: id must match the file name`);
  if (typeof c.name !== "string" || !c.name) out.push(`connections/${file}: needs a name`);
  if (!(CONNECTION_KINDS as readonly unknown[]).includes(c.kind)) out.push(`connections/${file}: kind is vault, issuer, portal or platform`);
  if (c.kind === "platform" && Array.isArray(c.holds) && c.holds.length) out.push(`connections/${file}: a learning platform holds no documents`);
  for (const h of Array.isArray(c.holds) ? c.holds : []) if (!documentType(h)) out.push(`connections/${file}: holds an unknown document type ${h}`);
  return out;
}

export type ChecklistRow = { type: DocumentType; custody: Custody | null; where?: string; connection?: string; heldBy: string[] };

/** The checklist: every document type, what the person said, and which connections hold it. */
export function checklist(cards: readonly Partial<CustodyCard>[], connections: readonly Connection[]): ChecklistRow[] {
  return DOCUMENT_TYPES.map((type) => {
    const c = cards.find((x) => x.type === type.id);
    return {
      type,
      custody: c?.custody && isCustody(c.custody) ? c.custody : null,
      where: c?.where,
      connection: c?.connection,
      heldBy: connections.filter((k) => k.holds.includes(type.id)).map((k) => k.id),
    };
  });
}

/** The folder host's documents: read the cards and connections, check one off, add a connection. */
export function filesDocuments(root: string, paths: RecordPaths, now: () => Date = () => new Date()) {
  async function list(dir: string, ext: string): Promise<string[]> {
    try {
      return (await readdir(join(root, dir))).filter((n) => n.endsWith(ext) && n.toLowerCase() !== "readme.md").sort();
    } catch {
      return [];
    }
  }
  async function connections(): Promise<Connection[]> {
    const out: Connection[] = [];
    for (const n of await list(paths.connections, ".json")) out.push(JSON.parse(await readFile(join(root, paths.connections, n), "utf8")) as Connection);
    return out;
  }
  async function cards(): Promise<Partial<CustodyCard>[]> {
    const out: Partial<CustodyCard>[] = [];
    for (const n of await list(paths.documents, ".md")) out.push(parseCard(await readFile(join(root, paths.documents, n), "utf8")).card);
    return out;
  }
  return {
    connections,
    cards,
    async checklist() {
      return checklist(await cards(), await connections());
    },
    async check(typeId: string, custody: Custody, opts: { where?: string; connection?: string } = {}) {
      if (!documentType(typeId)) throw new Error(`no such document type: ${typeId}; \`reflect documents\` lists them`);
      if (custody === "connection") {
        if (!opts.connection) throw new Error("custody connection needs --connection <id>");
        const k = (await connections()).find((x) => x.id === opts.connection);
        if (!k) throw new Error(`no such connection: ${opts.connection}`);
        if (!k.holds.includes(typeId)) throw new Error(`${k.name} does not hold ${typeId}; add it to the connection first`);
      } else if (opts.connection) throw new Error("--connection goes only with custody connection");
      const rel = join(paths.documents, `${typeId}.md`);
      const old = await readRecord(root, rel);
      const card: CustodyCard = { type: typeId, custody, where: opts.where?.trim() || undefined, connection: opts.connection, checkedAt: now().toISOString() };
      const history = [...(old ? parseCard(old).history : []), historyLine(card)];
      await writeRecord(root, rel, cardText(card, history), old ? "append" : "create", card.checkedAt);
      return card;
    },
    async connect(id: string, name: string, kind: Connection["kind"], holds: string[]) {
      if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(id)) throw new Error("a connection id is lowercase letters, digits and dashes");
      const rel = join(paths.connections, `${id}.json`);
      const old = await readRecord(root, rel);
      const prev = old ? (JSON.parse(old) as Connection) : null;
      const c: Connection = { id, name, kind, holds: [...new Set([...(prev?.holds ?? []), ...holds])].sort(), created_at: prev?.created_at ?? now().toISOString() };
      const problems = connectionProblems(c, `${id}.json`);
      if (problems.length) throw new Error(problems.join("; "));
      await writeRecord(root, rel, `${JSON.stringify(c, null, 2)}\n`, old ? "append" : "create", now().toISOString());
      return c;
    },
  };
}

/** Problems in an instance's documents and connections, for `reflect check`. */
export async function documentProblems(root: string, paths: RecordPaths): Promise<string[]> {
  const out: string[] = [];
  const names = await readdir(join(root, paths.connections)).catch(() => [] as string[]);
  const conns: Connection[] = [];
  for (const n of names.filter((x) => x.endsWith(".json"))) {
    try {
      const c = JSON.parse(await readFile(join(root, paths.connections, n), "utf8")) as Connection;
      out.push(...connectionProblems(c, n));
      conns.push(c);
    } catch {
      out.push(`connections/${n}: not JSON`);
    }
  }
  const cardNames = (await readdir(join(root, paths.documents)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md");
  for (const n of cardNames) {
    const { card } = parseCard(await readFile(join(root, paths.documents, n), "utf8"));
    if (!card.type || `${card.type}.md` !== n) out.push(`documents/${n}: type must match the file name`);
    else if (!documentType(card.type)) out.push(`documents/${n}: unknown document type ${card.type}`);
    if (!card.custody || !isCustody(card.custody)) out.push(`documents/${n}: custody is one of ${CUSTODY.join(", ")}`);
    if (card.custody === "connection" && !conns.some((k) => k.id === card.connection && k.holds.includes(card.type ?? ""))) out.push(`documents/${n}: names a connection that does not hold it`);
  }
  return out;
}
