// A prompt the person gives to the AI they talk to most, so it can say what it already knows about
// them: facts, the story so far, beliefs, values and the question they keep turning over. A variation
// of the founder's puzzle. Its answer is shown to the person and nothing is kept unless they mark an
// item as their own, which makes it a statement of theirs like any other.

export const BIOGRAPHY_SECTIONS = [
  { id: "origins", title: "Origins", about: "where and to whom I was born" },
  { id: "childhood", title: "Childhood", about: "what growing up was like" },
  { id: "schooling", title: "Schooling", about: "schools, teachers, what I studied" },
  { id: "turning_points", title: "Turning points", about: "moments that changed my direction" },
  { id: "work", title: "Work", about: "what I have done and do now" },
  { id: "relationships", title: "Relationships", about: "people who shaped me, by role, without their private details" },
  { id: "places", title: "Places", about: "where I have lived" },
  { id: "now", title: "Now", about: "what my life is like at present" },
] as const;

export type SectionId = (typeof BIOGRAPHY_SECTIONS)[number]["id"];

/** The facts an AI may report, and the values the profile accepts for each. */
export const REPORTABLE_FACTS = ["name", "date_of_birth", "occupation", "marital_status", "children"] as const;
export type ReportableFact = (typeof REPORTABLE_FACTS)[number];

const MARITAL = ["single", "dating", "partnered", "married", "separated", "divorced", "widowed"];

export const PROFILE_PROMPT = `You know me from our past conversations and from anything you have saved about me. I keep a private profile in Reflect, a journal app, and I want to see what you already know about me: the facts of my life, its story so far, what I believe and value, and the question I keep turning over.

Work only from what you actually know about me: your saved memory, this chat, and any past chats you can see. Do not interview me first. Do not search the web.

Rules:
1. Never invent. If you do not know something, leave it out (null or an empty list) and add it to open_questions. A short, true profile is better than a full one.
2. For every entry, say where it comes from. basis is "i-said" when I told you this, and "inferred" when you concluded it from what we talked about. Use my own words when you have them.
3. For every entry, write one sentence of evidence: what in our conversations it rests on. Describe topics; do not quote other people. Leave out health details, exact money figures, and anyone else's private information.
4. Give a confidence (low, medium or high) for every entry.
5. Give a date or a year only when I told you it. Never estimate my age.
6. Describe; do not judge. Do not flatter me, diagnose me, or advise me.
7. The puzzle is one or two sentences phrased as a question I keep coming back to: not a job title and not a goal. If the pattern is thin, say so with low confidence, or leave it null.

Facts (each null unless you know it):
- name
- date_of_birth, as YYYY-MM-DD
- occupation, in my words
- marital_status, one of: ${MARITAL.join(", ")}
- children, a number

Biography sections (an empty list is fine):
${BIOGRAPHY_SECTIONS.map((s) => `- ${s.id}: ${s.about}`).join("\n")}

Reply with one JSON code block and nothing before it, in exactly this shape:

\`\`\`json
{
  "kind": "reflect.profile-source",
  "version": 1,
  "reported_by": { "assistant": "<your product name, e.g. ChatGPT or Claude>", "basis": "<what you drew on: saved memory, past chats, this chat>", "reported_at": "<today, YYYY-MM-DD>" },
  "facts": {
    "name": null,
    "date_of_birth": null,
    "occupation": { "value": "...", "basis": "i-said", "evidence": "...", "confidence": "high" },
    "marital_status": null,
    "children": null
  },
  "biography": [ { "section": "work", "text": "...", "when": "<only if I told you>", "basis": "i-said", "evidence": "...", "confidence": "medium" } ],
  "beliefs": [ { "text": "...", "basis": "i-said", "evidence": "...", "confidence": "high" } ],
  "values": [ { "text": "...", "basis": "inferred", "evidence": "...", "confidence": "low" } ],
  "puzzle": { "statement": "<the question I keep turning over>", "why": "<the pattern that points to it>", "confidence": "low", "alternatives": [] },
  "open_questions": ["<what you could not tell and I should answer>"]
}
\`\`\`

After the code block, in two or three sentences, tell me which part you are least sure of and ask me the one question that would most improve it. I will correct this before anything is kept as mine.`;

export type Basis = "i-said" | "inferred";
export type Confidence = "low" | "medium" | "high";
export type Sourced = { basis: Basis; evidence: string; confidence: Confidence };
export type ReportedFact = Sourced & { key: ReportableFact; value: string };
export type ReportedItem = Sourced & { text: string };
export type ReportedStory = ReportedItem & { section: SectionId; when: string | null };

export type ProfileReport = {
  assistant: string;
  drewOn: string;
  reportedAt: string | null;
  facts: ReportedFact[];
  biography: ReportedStory[];
  beliefs: ReportedItem[];
  values: ReportedItem[];
  puzzle: { statement: string; why: string; confidence: Confidence; alternatives: string[] } | null;
  openQuestions: string[];
  /** The text the AI wrote after the code block, shown as it wrote it. */
  note: string;
  /** Entries left out because they broke the shape, so the person knows something was dropped. */
  dropped: number;
};

const str = (v: unknown): string => (typeof v === "string" ? v.trim() : "");
const conf = (v: unknown): Confidence | null => (v === "low" || v === "medium" || v === "high" ? v : null);
const basis = (v: unknown): Basis | null => (v === "i-said" || v === "inferred" ? v : null);

function sourced(o: Record<string, unknown>): Sourced | null {
  const b = basis(o.basis);
  const c = conf(o.confidence);
  return b && c ? { basis: b, evidence: str(o.evidence), confidence: c } : null;
}

/** A fact value the profile can accept as stated, or null. */
export function factValue(key: ReportableFact, v: unknown): string | null {
  const s = typeof v === "number" ? String(v) : str(v);
  if (!s || s.length > 200) return null;
  if (key === "date_of_birth") return /^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(Date.parse(s)) ? s : null;
  if (key === "marital_status") return MARITAL.includes(s) ? s : null;
  if (key === "children") return /^\d+$/.test(s) && Number(s) <= 30 ? s : null;
  return s;
}

/** Read what the AI pasted back. Returns an error message when there is no usable report. */
export function parseProfileReply(reply: string): { report: ProfileReport } | { error: string } {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/.exec(reply);
  const body = fenced ? fenced[1]! : reply.slice(reply.indexOf("{"), reply.lastIndexOf("}") + 1);
  let raw: unknown;
  try {
    raw = JSON.parse(body);
  } catch {
    return { error: "That doesn't look like the reply to the prompt. Paste the whole answer, including the code block." };
  }
  const o = (raw ?? {}) as Record<string, unknown>;
  if (o.kind !== "reflect.profile-source") return { error: "That reply isn't for this prompt. Copy the prompt again and paste its answer." };
  let dropped = 0;
  const by = (o.reported_by ?? {}) as Record<string, unknown>;

  const facts: ReportedFact[] = [];
  const f = (o.facts ?? {}) as Record<string, unknown>;
  for (const key of REPORTABLE_FACTS) {
    const e = f[key];
    if (e === null || e === undefined) continue;
    const r = e as Record<string, unknown>;
    const s = sourced(r);
    const value = factValue(key, r.value);
    if (s && value) facts.push({ key, value, ...s });
    else dropped++;
  }

  const items = (list: unknown): ReportedItem[] =>
    (Array.isArray(list) ? list : []).flatMap((e) => {
      const r = (e ?? {}) as Record<string, unknown>;
      const s = sourced(r);
      const text = str(r.text);
      if (s && text) return [{ text, ...s }];
      dropped++;
      return [];
    });

  const biography: ReportedStory[] = (Array.isArray(o.biography) ? o.biography : []).flatMap((e) => {
    const r = (e ?? {}) as Record<string, unknown>;
    const section = BIOGRAPHY_SECTIONS.find((x) => x.id === r.section)?.id;
    const s = sourced(r);
    const text = str(r.text);
    if (section && s && text) return [{ section, text, when: str(r.when) || null, ...s }];
    dropped++;
    return [];
  });

  const p = o.puzzle as Record<string, unknown> | null | undefined;
  const statement = str(p?.statement);
  const pc = conf(p?.confidence);
  const puzzle =
    p && statement && pc
      ? { statement, why: str(p.why), confidence: pc, alternatives: (Array.isArray(p.alternatives) ? p.alternatives : []).map(str).filter(Boolean) }
      : null;
  if (p && !puzzle) dropped++;

  const reportedAt = str(by.reported_at);
  return {
    report: {
      assistant: str(by.assistant) || "Your AI",
      drewOn: str(by.basis),
      reportedAt: /^\d{4}-\d{2}-\d{2}$/.test(reportedAt) ? reportedAt : null,
      facts,
      biography,
      beliefs: items(o.beliefs),
      values: items(o.values),
      puzzle,
      openQuestions: (Array.isArray(o.open_questions) ? o.open_questions : []).map(str).filter(Boolean),
      note: fenced ? reply.slice(fenced.index + fenced[0].length).trim() : "",
      dropped,
    },
  };
}
