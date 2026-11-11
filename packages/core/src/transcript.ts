// A school transcript as courses the person can look back over. A model reads the transcript's text and
// drafts its rows (term, code, title, credits, grade); a row is kept as a draft only when its title, and
// its grade when it has one, are really in the text, so nothing is invented. The person confirms each
// row before it becomes a course. The grade is the school's; the person's own grasp of the course is a
// separate rating they give, the same four levels as for any subject. Validating a transcript is not
// needed and not built.
import { join } from "node:path";
import { keepSubject, slugOf } from "./learning.ts";
import { headOf } from "./lifetime.ts";
import { readRecord, writeRecord } from "./record-io.ts";

export const COURSES = join("learn", "courses");

export type CourseRow = { term: string; code?: string; title: string; credits?: number; grade?: string };
export type TranscriptDraft = { institution?: string; courses: CourseRow[]; dropped: string[] };
export type Chat = (system: string, text: string) => Promise<unknown | null>;

export const TRANSCRIPT_PROMPT = [
  "You read the text of an academic transcript and list its courses. Answer with JSON only:",
  '{"institution": string or null, "courses": [{"term": string, "code": string or null, "title": string, "credits": number or null, "grade": string or null}]}',
  "Rules: copy each title, code, term and grade exactly as the transcript writes them. One entry per course taken.",
  "Leave out totals, GPA lines, headers and anything that is not a course. Never guess a field; use null.",
].join("\n");

const squash = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);

/** Keep the rows whose title (and grade, when given) appear in the transcript's text. */
export function anchorCourses(text: string, answer: unknown): TranscriptDraft {
  const t = squash(text);
  const a = (answer ?? {}) as { institution?: unknown; courses?: unknown };
  const out: CourseRow[] = [];
  const dropped: string[] = [];
  for (const raw of Array.isArray(a.courses) ? a.courses : []) {
    const r = (raw ?? {}) as Record<string, unknown>;
    const title = str(r.title);
    if (!title) continue;
    const grade = str(r.grade);
    if (!t.includes(squash(title)) || (grade && !new RegExp(`(^|[^a-z0-9])${squash(grade).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}($|[^a-z0-9])`).test(t))) {
      dropped.push(title);
      continue;
    }
    const code = str(r.code);
    const credits = typeof r.credits === "number" && Number.isFinite(r.credits) && r.credits >= 0 && r.credits < 100 ? r.credits : undefined;
    out.push({ term: str(r.term) ?? "", title, ...(code && t.includes(squash(code)) ? { code } : {}), ...(credits !== undefined ? { credits } : {}), ...(grade ? { grade } : {}) });
  }
  const institution = str(a.institution);
  return { ...(institution && t.includes(squash(institution)) ? { institution } : {}), courses: out, dropped };
}

/** Draft the courses of a transcript's text, or null when no model is set up. */
export async function draftTranscript(text: string, chat: Chat): Promise<TranscriptDraft | null> {
  if (!text.trim()) throw new Error("the transcript has no text; a scanned image needs to be typed or read out first");
  const answer = await chat(TRANSCRIPT_PROMPT, text.slice(0, 60_000));
  return answer === null ? null : anchorCourses(text, answer);
}

export const courseId = (c: CourseRow) => slugOf(`${c.term} ${c.code ?? c.title}`);

/**
 * Keep a confirmed course as learn/courses/<term>-<code>.md, linked to a subject of the same title so the
 * person can rate their grasp of it like any subject. The course note keeps the school's record as given.
 */
export async function keepCourse(root: string, c: CourseRow, institution: string | undefined, now: Date): Promise<string> {
  if (!c.title.trim()) throw new Error("a course needs a title");
  const id = courseId(c);
  const rel = join(COURSES, `${id}.md`);
  const old = await readRecord(root, rel);
  const subject = await keepSubject(root, c.title, {}, now);
  const q = (s: string) => JSON.stringify(s);
  const head = [`course: ${id}`, `title: ${q(c.title)}`, `term: ${q(c.term)}`];
  if (c.code) head.push(`code: ${q(c.code)}`);
  if (c.credits !== undefined) head.push(`credits: ${c.credits}`);
  if (c.grade) head.push(`grade: ${q(c.grade)}`);
  if (institution) head.push(`institution: ${q(institution)}`);
  head.push(`subject: "[[learn/subjects/${subject.id}]]"`);
  const at = now.toISOString();
  await writeRecord(root, rel, `---\n${head.join("\n")}\n---\n\n# ${c.title}\n\n- ${at.slice(0, 10)} kept from a transcript\n`, old ? "append" : "create", at);
  return rel;
}

export type Course = { id: string; title: string; term: string; code?: string; credits?: number; grade?: string; institution?: string; subject: string };

export function parseCourse(id: string, text: string): Course {
  const h = headOf(text);
  return {
    id,
    title: h.title ?? id,
    term: h.term ?? "",
    ...(h.code ? { code: h.code } : {}),
    ...(h.credits ? { credits: Number(h.credits) } : {}),
    ...(h.grade ? { grade: h.grade } : {}),
    ...(h.institution ? { institution: h.institution } : {}),
    subject: /\[\[learn\/subjects\/([^\]]+)\]\]/.exec(h.subject ?? "")?.[1] ?? slugOf(h.title ?? id),
  };
}

export async function listCourses(root: string): Promise<Course[]> {
  const { readdir, readFile } = await import("node:fs/promises");
  const names = (await readdir(join(root, COURSES)).catch(() => [] as string[])).filter((n) => n.endsWith(".md") && n.toLowerCase() !== "readme.md").sort();
  return Promise.all(names.map(async (n) => parseCourse(n.slice(0, -3), await readFile(join(root, COURSES, n), "utf8"))));
}

/** The text of a plain-text transcript. A PDF is read by its host: the CLI and the app each use unpdf. */
export function transcriptText(bytes: Uint8Array, name: string): string {
  if (/\.pdf$/i.test(name)) throw new Error("a PDF is read by the host, not the core");
  return new TextDecoder().decode(bytes);
}
