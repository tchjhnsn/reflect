// The memory sweep: a prompt the person pastes into the AI they talk to most, which answers from what it
// already knows about them with the books, films, talks, links, conversations, courses and credentials that
// came up, the subjects they showed knowledge of, subjects it suggests, and the person's own ideas with the
// inputs it saw them draw on. Every item says whether the person said it or the AI inferred it, with one
// sentence of evidence and a confidence. The reply is a draft: the person keeps items one by one, and
// nothing in it sets a grasp. An idea's input the AI only inferred is kept as `maybe from`.
import { keepCredential } from "./credentials.ts";
import { keepIdea, resolveTarget, type Target } from "./ideas.ts";
import { keepBook, keepConversation, keepFilm, keepResource, keepSubject, keepTalk, sourceOf, type BookStatus, type FilmStatus } from "./learning.ts";
import { keepCourse } from "./transcript.ts";

export const SWEEP_PROMPT = `You know me from our past conversations and from anything you have saved about me. Use that to list what I have read, watched, heard and saved, the subjects I know, and the ideas of my own I have been working on, for Reflect (my own notes app). Work only from what you actually know about me: your saved memory, this chat, and past chats you can see. Do not interview me first. Do not search the web.

Rules:
1. Never invent. If you are unsure a title or author is right, say so in evidence and use low confidence. Do not fill in authors, years or links you do not know.
2. For every item, say where it comes from. basis is "i-said" when I told you, and "you-inferred" when you concluded it from what we talked about.
3. For every item, one sentence of evidence: what in our conversations it rests on. Describe topics; do not quote other people. Leave out health details, exact money figures and anyone else's private information.
4. For books and films, say how I engaged: "finished", "started", "referenced" (I cited or mentioned it) or "wants" (I said I want to read or watch it). Do not upgrade a mention to "finished".
5. Inputs and outputs are different things. Books, films, talks (a person I saw speak), links, conversations, courses and subjects are inputs: what I took in. My own framings, theses and hybrids (for example "X applied to Y") are outputs: put them in ideas, never in subjects. For each idea, list the inputs you saw me draw on, by title, and only those.
6. Subjects are established fields I study, in two lists that must never be mixed:
   - shown: subjects where I displayed knowledge in our conversations: I explained it, used it to solve something, corrected you on it, or taught it to someone.
   - suggested: subjects you think connect to what I read or care about, where I have not shown knowledge. Interest is not knowledge.
   Do not rate how well I know anything. I rate that myself.
7. Links: articles, posts and threads I shared or referred to (X, Substack, Reddit, YouTube, blogs, papers). Include the url only if you actually have it.
8. Do not flatter me. A thin list is fine.

Reply with one JSON code block and nothing before it, in exactly this shape:

\`\`\`json
{
  "kind": "reflect.learn-source",
  "version": 1,
  "reported_by": { "assistant": "<your product name>", "basis": "<saved memory, past chats, this chat>", "reported_at": "<YYYY-MM-DD>" },
  "books": [ { "title": "...", "authors": ["..."], "engaged": "finished|started|referenced|wants", "subjects": ["..."], "basis": "i-said|you-inferred", "evidence": "...", "confidence": "low|medium|high" } ],
  "films": [ { "title": "...", "year": null, "kind": "film|documentary|series|short", "engaged": "finished|started|referenced|wants", "subjects": ["..."], "basis": "...", "evidence": "...", "confidence": "..." } ],
  "talks": [ { "title": "...", "speaker": "...", "url": null, "subjects": ["..."], "basis": "...", "evidence": "...", "confidence": "..." } ],
  "links": [ { "title": "...", "url": null, "kind": "post|thread|article|video|newsletter|paper|podcast", "by": null, "subjects": ["..."], "basis": "...", "evidence": "...", "confidence": "..." } ],
  "conversations": [ { "with": ["<who, as I named them>"], "about": "...", "when": null, "basis": "...", "evidence": "...", "confidence": "..." } ],
  "courses": [ { "title": "...", "where": "<school or platform>", "basis": "...", "evidence": "...", "confidence": "..." } ],
  "credentials": [ { "name": "...", "issuer": "...", "basis": "...", "evidence": "...", "confidence": "..." } ],
  "subjects": {
    "shown": [ { "name": "...", "how": "explained|applied|corrected|taught", "evidence": "...", "confidence": "..." } ],
    "suggested": [ { "name": "...", "because": "<which inputs or interests point to it>", "confidence": "..." } ]
  },
  "ideas": [ { "title": "<a short name>", "statement": "<the idea, in my words where you have them>", "inputs": [ { "title": "<an input from above>", "how": "<how it fed the idea>" } ], "basis": "...", "evidence": "...", "confidence": "..." } ],
  "open_questions": ["<what you could not tell>"]
}
\`\`\`

After the code block, in two or three sentences, tell me which items you are least sure of. I will correct this before anything is kept.`;

type Base = { basis?: string; evidence?: string; confidence?: string };
export type SweepItem =
  | ({ kind: "book"; title: string; authors: string[]; engaged: string; subjects: string[] } & Base)
  | ({ kind: "film"; title: string; year?: number; film: string; engaged: string; subjects: string[] } & Base)
  | ({ kind: "talk"; title: string; speaker: string; url?: string; subjects: string[] } & Base)
  | ({ kind: "link"; title: string; url?: string; link: string; by?: string; subjects: string[] } & Base)
  | ({ kind: "conversation"; with: string[]; about: string; when?: string } & Base)
  | ({ kind: "course"; title: string; where?: string } & Base)
  | ({ kind: "credential"; name: string; issuer: string } & Base)
  | ({ kind: "shown"; name: string; how?: string } & Base)
  | ({ kind: "suggested"; name: string; because?: string } & Base)
  | ({ kind: "idea"; title: string; statement: string; inputs: { title: string; how?: string }[] } & Base);
export type Sweep = { assistant: string; reportedAt?: string; items: SweepItem[]; openQuestions: string[] };

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const strs = (v: unknown) => (Array.isArray(v) ? v.map(str).filter((s): s is string => !!s) : []);
const arr = (v: unknown) => (Array.isArray(v) ? (v as Record<string, unknown>[]).filter((x) => x && typeof x === "object") : []);
const base = (r: Record<string, unknown>): Base => ({ ...(str(r.basis) ? { basis: str(r.basis) } : {}), ...(str(r.evidence) ? { evidence: str(r.evidence) } : {}), ...(str(r.confidence) ? { confidence: str(r.confidence) } : {}) });

/** Read the AI's reply: the JSON alone, or the first JSON code block in its message. */
export function readSweep(reply: string): Sweep {
  const fenced = /```(?:json)?\s*\n([\s\S]*?)\n```/.exec(reply);
  let j: Record<string, unknown>;
  try {
    j = JSON.parse((fenced ? fenced[1]! : reply).trim()) as Record<string, unknown>;
  } catch {
    throw new Error("the reply has no JSON the sweep can read; paste the whole code block");
  }
  if (j.kind !== "reflect.learn-source" || j.version !== 1) throw new Error('the JSON is not a memory sweep (kind "reflect.learn-source", version 1)');
  const items: SweepItem[] = [];
  for (const r of arr(j.books)) if (str(r.title)) items.push({ kind: "book", title: str(r.title)!, authors: strs(r.authors), engaged: str(r.engaged) ?? "referenced", subjects: strs(r.subjects), ...base(r) });
  for (const r of arr(j.films)) if (str(r.title)) items.push({ kind: "film", title: str(r.title)!, ...(typeof r.year === "number" && Number.isInteger(r.year) ? { year: r.year } : {}), film: str(r.kind) ?? "film", engaged: str(r.engaged) ?? "referenced", subjects: strs(r.subjects), ...base(r) });
  for (const r of arr(j.talks)) if (str(r.title) && str(r.speaker)) items.push({ kind: "talk", title: str(r.title)!, speaker: str(r.speaker)!, ...(str(r.url) ? { url: str(r.url) } : {}), subjects: strs(r.subjects), ...base(r) });
  for (const r of arr(j.links)) if (str(r.title) || str(r.url)) items.push({ kind: "link", title: str(r.title) ?? str(r.url)!, ...(str(r.url) ? { url: str(r.url) } : {}), link: str(r.kind) ?? "article", ...(str(r.by) ? { by: str(r.by) } : {}), subjects: strs(r.subjects), ...base(r) });
  for (const r of arr(j.conversations)) if (strs(r.with).length && str(r.about)) items.push({ kind: "conversation", with: strs(r.with), about: str(r.about)!, ...(str(r.when) ? { when: str(r.when) } : {}), ...base(r) });
  for (const r of arr(j.courses)) if (str(r.title)) items.push({ kind: "course", title: str(r.title)!, ...(str(r.where) ? { where: str(r.where) } : {}), ...base(r) });
  for (const r of arr(j.credentials)) if (str(r.name) && str(r.issuer)) items.push({ kind: "credential", name: str(r.name)!, issuer: str(r.issuer)!, ...base(r) });
  const subj = (j.subjects ?? {}) as Record<string, unknown>;
  for (const r of arr(subj.shown)) if (str(r.name)) items.push({ kind: "shown", name: str(r.name)!, ...(str(r.how) ? { how: str(r.how) } : {}), ...base(r) });
  for (const r of arr(subj.suggested)) if (str(r.name)) items.push({ kind: "suggested", name: str(r.name)!, ...(str(r.because) ? { because: str(r.because) } : {}), ...base(r) });
  for (const r of arr(j.ideas)) if (str(r.title) && str(r.statement)) items.push({ kind: "idea", title: str(r.title)!, statement: str(r.statement)!, inputs: arr(r.inputs).filter((x) => str(x.title)).map((x) => ({ title: str(x.title)!, ...(str(x.how) ? { how: str(x.how) } : {}) })), ...base(r) });
  const by = (j.reported_by ?? {}) as Record<string, unknown>;
  return { assistant: str(by.assistant) ?? "your AI", ...(str(by.reported_at) ? { reportedAt: str(by.reported_at) } : {}), items, openQuestions: strs(j.open_questions) };
}

/** One line per item, for the person to check before keeping. */
export function sweepLine(i: SweepItem): string {
  const tail = `${i.basis === "i-said" ? "you said" : "it inferred"}${i.confidence ? `, ${i.confidence}` : ""}${i.evidence ? `: ${i.evidence}` : ""}`;
  switch (i.kind) {
    case "book": return `book: ${i.title}${i.authors.length ? ` by ${i.authors.join(", ")}` : ""} (${i.engaged}); ${tail}`;
    case "film": return `${i.film}: ${i.title}${i.year ? ` (${i.year})` : ""} (${i.engaged}); ${tail}`;
    case "talk": return `talk: ${i.title} by ${i.speaker}; ${tail}`;
    case "link": return `${i.link}: ${i.title}${i.url ? ` <${i.url}>` : ""}; ${tail}`;
    case "conversation": return `conversation with ${i.with.join(", ")} about ${i.about}; ${tail}`;
    case "course": return `course: ${i.title}${i.where ? ` at ${i.where}` : ""}; ${tail}`;
    case "credential": return `credential: ${i.name} from ${i.issuer}; ${tail}`;
    case "shown": return `subject you showed knowledge of: ${i.name}${i.how ? ` (${i.how})` : ""}; ${tail}`;
    case "suggested": return `subject it suggests: ${i.name}${i.because ? `, because ${i.because}` : ""}`;
    case "idea": return `your idea: ${i.title}: ${i.statement}${i.inputs.length ? `; from ${i.inputs.map((x) => x.title).join(", ")}` : ""}`;
  }
}

const BOOK: Record<string, BookStatus> = { finished: "read", started: "reading", referenced: "referenced", wants: "want" };
const FILM: Record<string, FilmStatus> = { finished: "watched", started: "watching", referenced: "referenced", wants: "want" };
const LINK_KINDS = new Set(["post", "thread", "article", "video", "newsletter", "paper", "podcast"]);

/** Keep the items the person chose (numbers from 1), in the order given. Ideas come last so their inputs are found. */
export async function keepSweep(root: string, s: Sweep, numbers: readonly number[], now: Date): Promise<string[]> {
  const chosen = [...new Set(numbers)].map((n) => {
    const i = s.items[n - 1];
    if (!i) throw new Error(`no item ${n}; the draft has ${s.items.length}`);
    return i;
  });
  const source = (i: SweepItem) => `${s.assistant}${s.reportedAt ? ` ${s.reportedAt}` : ""}, ${i.basis === "i-said" ? "you said" : "it inferred"}${i.evidence ? `: ${i.evidence}` : ""}`;
  const kept: string[] = [];
  for (const i of [...chosen.filter((x) => x.kind !== "idea"), ...chosen.filter((x) => x.kind === "idea")]) {
    if (i.kind === "book") kept.push(`learn/books/${(await keepBook(root, i.title, { authors: i.authors, status: BOOK[i.engaged] ?? "referenced", subjects: i.subjects, source: source(i) }, now)).id}.md`);
    else if (i.kind === "film") kept.push(`learn/films/${(await keepFilm(root, i.title, { kind: ["film", "documentary", "series", "short"].includes(i.film) ? i.film : "film", ...(i.year ? { year: i.year } : {}), status: FILM[i.engaged] ?? "referenced", subjects: i.subjects, source: source(i) }, now)).id}.md`);
    else if (i.kind === "talk") kept.push(`learn/talks/${(await keepTalk(root, i.title, { speaker: i.speaker, ...(i.url && /^https?:\/\//.test(i.url) ? { url: i.url } : {}), subjects: i.subjects, source: source(i) }, now)).id}.md`);
    else if (i.kind === "link") {
      const url = i.url && /^https?:\/\/\S+$/.test(i.url) ? i.url : undefined;
      const kind = LINK_KINDS.has(i.link) ? i.link : url && ["x", "reddit"].includes(sourceOf(url) ?? "") ? "post" : "article";
      kept.push(`learn/resources/${(await keepResource(root, i.title, { kind, ...(url ? { url } : {}), ...(i.by ? { by: i.by } : {}), subjects: i.subjects, source: source(i) }, now)).id}.md`);
    } else if (i.kind === "conversation") {
      const on = i.when && /^\d{4}(-\d{2}(-\d{2})?)?$/.test(i.when) ? i.when : undefined;
      kept.push(`learn/conversations/${(await keepConversation(root, { with: i.with, ...(on ? { on } : {}), about: i.about, source: source(i) }, now)).id}.md`);
    } else if (i.kind === "course") kept.push(await keepCourse(root, { term: "", title: i.title }, i.where, now));
    else if (i.kind === "credential") kept.push(`learn/credentials/${(await keepCredential(root, i.name, { issuer: i.issuer }, now)).id}.md`);
    else if (i.kind === "shown") kept.push(`learn/subjects/${(await keepSubject(root, i.name, { shown: { by: s.assistant, evidence: `${i.how ? `${i.how}: ` : ""}${i.evidence ?? "it saw you show it"}` } }, now)).id}.md`);
    else if (i.kind === "suggested") kept.push(`learn/subjects/${(await keepSubject(root, i.name, { suggested: { by: s.assistant, ...(i.because ? { because: i.because } : {}) } }, now)).id}.md`);
    else {
      const inputs: { target: Target; how?: string }[] = [];
      for (const x of i.inputs) inputs.push({ target: await resolveTarget(root, x.title), ...(x.how ? { how: x.how } : {}) });
      const said = i.basis === "i-said";
      kept.push(`ideas/${(await keepIdea(root, i.title, { statement: i.statement, ...(said ? { from: inputs } : { maybe: inputs }) }, now)).id}.md`);
    }
  }
  return kept;
}
