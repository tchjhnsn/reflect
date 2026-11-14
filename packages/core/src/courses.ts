/**
 * Taking a course inside Reflect. Open Yale Courses and MIT OpenCourseWare publish each lecture with its video, and
 * Yale with its audio and a transcript in chapters, MIT with captions. These read their public pages into a course's
 * lectures and a lecture's parts, so Reflect plays them from the publisher's own sources and copies nothing. Both are
 * CC BY-NC-SA: Reflect credits them and keeps no copy. Fetching is passed in, so it runs anywhere.
 */
import { readTranscript } from "./media";

export type CourseSite = "oyc" | "ocw";
export type LectureRef = { n: number; title: string; url: string; youtube: string | null };
export type Passage = { at: number | null; heading: boolean; who: string | null; text: string };
export type Lecture = { title: string; overview: string | null; youtube: string | null; audio: string | null; video: string | null; captions: string | null; transcript: Passage[] };
export type Fetch = (url: string) => Promise<string | null>;

export const COURSE_SITES: Record<CourseSite, { name: string; license: string; licenseUrl: string }> = {
  oyc: { name: "Open Yale Courses", license: "CC BY-NC-SA 3.0", licenseUrl: "https://creativecommons.org/licenses/by-nc-sa/3.0/us/" },
  ocw: { name: "MIT OpenCourseWare", license: "CC BY-NC-SA 4.0", licenseUrl: "https://creativecommons.org/licenses/by-nc-sa/4.0/" },
};

/** The publisher and canonical address of a course page, or null when Reflect cannot play it. */
export function courseSite(raw: string | null | undefined): { site: CourseSite; url: string } | null {
  if (!raw) return null;
  let u: URL;
  try { u = new URL(raw); } catch { return null; }
  if (u.protocol !== "https:" && u.protocol !== "http:") return null;
  const host = u.hostname.replace(/^www\./, "");
  const parts = u.pathname.split("/").filter(Boolean);
  if (host === "oyc.yale.edu" && parts.length >= 2 && /^[a-z]+-\d+[a-z]?(?:-\d+)?$/i.test(parts[1]!)) return { site: "oyc", url: `https://oyc.yale.edu/${parts[0]}/${parts[1]}` };
  if (host === "ocw.mit.edu" && parts[0] === "courses" && parts[1]) return { site: "ocw", url: `https://ocw.mit.edu/courses/${parts[1]}/` };
  return null;
}

const ENT: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", rsquo: "’", lsquo: "‘", ldquo: "“", rdquo: "”", mdash: "—", ndash: "–", hellip: "…" };
export function plain(html: string): string {
  return html
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e: string) => (e[0] === "#" ? String.fromCodePoint(e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : Number(e.slice(1))) : ENT[e.toLowerCase()] ?? m))
    .replace(/\s+/g, " ")
    .trim();
}
const abs = (href: string, origin: string) => { try { return new URL(href, origin).toString(); } catch { return null; } };
const hms = (s: string) => { const p = s.split(":").map(Number); return p.reduce((n, x) => n * 60 + x, 0); };

/** Open Yale: the lectures listed on a course page, in order. */
export function oycLectures(html: string, courseUrl: string): LectureRef[] {
  const out: LectureRef[] = [];
  const seen = new Set<string>();
  for (const m of html.matchAll(/session-display-title"><a href="(\/[^"]+\/lecture-(\d+))"[^>]*>([\s\S]*?)<\/a>/g)) {
    if (seen.has(m[1]!)) continue;
    seen.add(m[1]!);
    out.push({ n: Number(m[2]), title: plain(m[3]!), url: abs(m[1]!, courseUrl)!, youtube: null });
  }
  return out.sort((a, b) => a.n - b.n);
}

/** Open Yale: a lecture page's video, audio, overview and transcript (chapters with their times, then paragraphs). */
export function oycLecture(html: string, pageUrl: string): Lecture {
  const title = plain(/session-full__title">([\s\S]*?)<\/h2>/.exec(html)?.[1] ?? "");
  const overview = /session-full__body-label">\s*Overview\s*<\/h3>\s*<p>([\s\S]*?)<\/p>/.exec(html)?.[1];
  const youtube = /youtube(?:-nocookie)?\.com\/embed\/([\w-]{11})/.exec(html)?.[1] ?? null;
  const mp3 = /href="([^"]+\.mp3)"/.exec(html)?.[1];
  const start = html.indexOf('id="transcript-top"');
  const transcript: Passage[] = [];
  if (start >= 0) {
    const body = html.slice(start, html.indexOf("</td>", start) > 0 ? html.indexOf("</td>", start) : undefined);
    for (const m of body.matchAll(/<h3>([\s\S]*?)<\/h3>|<p>((?:(?!<p>|<h3>)[\s\S])*?)<\/p>/g)) {
      if (m[1] !== undefined) {
        const h = plain(m[1]);
        const t = /\[(\d{1,2}:\d{2}:\d{2})\]\s*$/.exec(h);
        transcript.push({ at: t ? hms(t[1]!) : null, heading: true, who: null, text: t ? h.slice(0, t.index).trim() : h });
      } else if (m[2]) {
        const who = /^\s*<strong>([\s\S]*?):?\s*<\/strong>/.exec(m[2]);
        const text = plain(who ? m[2].slice(who[0].length) : m[2]).replace(/^:\s*/, "");
        if (text) transcript.push({ at: null, heading: false, who: who ? plain(who[1]!).replace(/:$/, "") : null, text });
      }
    }
  }
  return { title, overview: overview ? plain(overview) : null, youtube, audio: mp3 ? abs(mp3, pageUrl) : null, video: null, captions: null, transcript };
}

/** MIT OpenCourseWare: the address of a course's video gallery, from its home page. */
export function ocwGallery(html: string, courseUrl: string): string | null {
  const href = /href="([^"]*\/video_galleries\/[^"]+)"/.exec(html)?.[1];
  return href ? abs(href, courseUrl) : null;
}

/** MIT OpenCourseWare: the lectures in a video gallery, in order. */
export function ocwLectures(html: string, galleryUrl: string): LectureRef[] {
  const out: LectureRef[] = [];
  for (const m of html.matchAll(/<a class="video-link" href="([^"]+)">([\s\S]*?)<\/a>/g)) {
    const title = plain(/video-title">([\s\S]*?)<\/h5>/.exec(m[2]!)?.[1] ?? "");
    if (!title) continue;
    out.push({ n: out.length + 1, title, url: abs(m[1]!, galleryUrl)!, youtube: /\/vi\/([\w-]{11})\//.exec(m[2]!)?.[1] ?? null });
  }
  return out;
}

/** MIT OpenCourseWare: a lecture page's video, captions and the Internet Archive copy. */
export function ocwLecture(html: string, pageUrl: string): Lecture {
  const title = plain(/<h2[^>]*>([\s\S]*?)<\/h2>/.exec(html)?.[1] ?? /<title>([\s\S]*?)(?:\||<\/title>)/.exec(html)?.[1] ?? "");
  const youtube = /youtube(?:-nocookie)?\.com\/embed\/([\w-]{11})/.exec(html)?.[1] ?? null;
  const vtt = /(?:href|src)="([^"]+\.vtt)"/.exec(html)?.[1] ?? /["']([^"']+\.vtt)["']/.exec(html)?.[1];
  const mp4 = /(https?:\/\/(?:www\.)?archive\.org\/download\/[^"'\s]+\.mp4)/.exec(html)?.[1];
  const desc = /<meta name="description" content="([^"]*)"/.exec(html)?.[1];
  return { title, overview: desc ? plain(desc) : null, youtube, audio: null, video: mp4 ? mp4.replace(/^http:/, "https:") : null, captions: vtt ? abs(vtt, pageUrl) : null, transcript: [] };
}

/** Captions read as passages of about half a minute each, so they read as text. */
export function passagesFromCaptions(vtt: string, span = 30): Passage[] {
  const out: Passage[] = [];
  for (const p of readTranscript(vtt)) {
    const last = out.at(-1);
    if (last && last.at !== null && p.at - last.at < span) last.text += ` ${p.text}`;
    else out.push({ at: p.at, heading: false, who: null, text: p.text });
  }
  return out;
}

/** A course's lectures, read from its publisher. */
export async function courseLectures(raw: string, f: Fetch): Promise<{ site: CourseSite; url: string; lectures: LectureRef[] } | null> {
  const c = courseSite(raw);
  if (!c) return null;
  const home = await f(c.url);
  if (!home) return { ...c, lectures: [] };
  if (c.site === "oyc") return { ...c, lectures: oycLectures(home, c.url) };
  for (const gallery of [ocwGallery(home, c.url), `${c.url}video_galleries/video-lectures/`, `${c.url}video_galleries/lecture-videos/`]) {
    const g = gallery ? await f(gallery) : null;
    const lectures = g ? ocwLectures(g, gallery!) : [];
    if (lectures.length) return { ...c, lectures };
  }
  return { ...c, lectures: [] };
}

/** One lecture, with its transcript from the publisher's transcript or captions. */
export async function courseLecture(site: CourseSite, ref: LectureRef, f: Fetch): Promise<Lecture | null> {
  const html = await f(ref.url);
  if (!html) return null;
  if (site === "oyc") return oycLecture(html, ref.url);
  const l = ocwLecture(html, ref.url);
  const youtube = l.youtube ?? ref.youtube;
  const vtt = l.captions ? await f(l.captions) : null;
  return { ...l, title: ref.title, overview: null, youtube, transcript: vtt ? passagesFromCaptions(vtt) : [] };
}

/** The passage being spoken at a moment: the last timed passage at or before it. */
export function passageAt(transcript: readonly Passage[], s: number): number {
  let at = -1;
  transcript.forEach((p, i) => { if (p.at !== null && p.at <= s) at = i; });
  return at;
}

export type SyllabusPart = { heading: string; lines: string[] };
export type CourseInfo = { site: CourseSite; url: string; title: string; by: string | null; image: string | null; about: string[]; syllabus: SyllabusPart[]; lectures: LectureRef[] };

/** An HTML fragment as lines of plain text: one per paragraph, line break or list item. */
export function linesOf(html: string): string[] {
  return html.split(/<\/p>|<br\s*\/?>|<\/li>|<\/h\d>/i).map(plain).filter((l) => l && l !== " ");
}

/** Open Yale: a course page's title, professor, description and syllabus (structure, texts, requirements, grading). */
export function oycCourse(html: string): Omit<CourseInfo, "site" | "url" | "lectures"> {
  const fields: { label: string; html: string }[] = [];
  for (const m of html.matchAll(/views-label-[\w-]+">([\s\S]*?)<\/span>\s*<div class="field-content">([\s\S]*?)<\/div>\s*<\/div>/g)) fields.push({ label: plain(m[1]!), html: m[2]! });
  const pick = (l: string) => fields.find((f) => f.label === l);
  const title = plain(/<title>([\s\S]*?)(?:\||<\/title>)/.exec(html)?.[1] ?? "");
  const prof = pick("Professor");
  const syllabus: SyllabusPart[] = [];
  const seen = new Set<string>();
  for (const f of fields) {
    if (["Course Number", "About the Course", "Description", "Professor"].includes(f.label) || seen.has(f.label)) continue;
    seen.add(f.label);
    const lines = linesOf(f.html).filter((l) => !/click here|^The Open Yale Courses Series$/i.test(l));
    if (lines.length) syllabus.push({ heading: f.label, lines });
  }
  return { title, by: prof ? plain(prof.html) : null, image: null, about: linesOf((pick("About the Course") ?? pick("Description"))?.html ?? ""), syllabus };
}

/** MIT OpenCourseWare: a course home page's title, picture and description. */
export function ocwCourse(html: string): Omit<CourseInfo, "site" | "url" | "lectures" | "syllabus"> {
  const title = plain(/<h1>\s*<a[^>]*>([\s\S]*?)<\/a>/.exec(html)?.[1] ?? /<title>([\s\S]*?)(?:\||<\/title>)/.exec(html)?.[1] ?? "");
  const image = /property="og:image" content="(https:[^"]+)"/.exec(html)?.[1] ?? null;
  const about = /id="full-description"[^>]*>([\s\S]*?)<\/div>/.exec(html)?.[1];
  return { title, by: null, image, about: about ? linesOf(about) : [] };
}

/** MIT OpenCourseWare: a syllabus page's sections, each heading with its lines. */
export function ocwSyllabus(html: string): SyllabusPart[] {
  const main = /<main id="course-content-section">([\s\S]*?)<\/main>/.exec(html)?.[1];
  if (!main) return [];
  const out: SyllabusPart[] = [];
  for (const part of main.split(/(?=<h[23][^>]*>)/)) {
    const h = /^<h[23][^>]*>([\s\S]*?)<\/h[23]>/.exec(part);
    const lines = linesOf(h ? part.slice(h[0].length) : part).slice(0, 40);
    if (h || lines.length) out.push({ heading: h ? plain(h[1]!) : "Syllabus", lines });
  }
  return out.filter((p) => p.lines.length);
}

/** Everything to explore a course before taking it: its description, syllabus and lectures. */
export async function courseInfo(raw: string, f: Fetch): Promise<CourseInfo | null> {
  const c = courseSite(raw);
  if (!c) return null;
  const [home, lectures] = await Promise.all([f(c.url), courseLectures(c.url, f)]);
  if (!home) return null;
  if (c.site === "oyc") return { ...c, ...oycCourse(home), lectures: lectures?.lectures ?? [] };
  const syl = await f(`${c.url}pages/syllabus/`);
  return { ...c, ...ocwCourse(home), syllabus: syl ? ocwSyllabus(syl) : [], lectures: lectures?.lectures ?? [] };
}

/** A syllabus section that lists books to read: Texts, Readings, Textbook, Books. */
export const isTextsHeading = (h: string): boolean => /^(required\s+)?(texts?|readings?|textbooks?|books?)\b/i.test(h.replace(/:$/, "").trim());

export type TextCitation = { title: string; author: string | null; isbn: string | null };

/**
 * One line of a syllabus's texts read as a book to find: "Plato, Phaedo", "Browning, Christopher. Ordinary Men. New
 * York: ...", a chapter "Barnes, Julian. "The Dream." In History of the World..." (the book it is in), or any line with
 * an ISBN. Null for a heading such as "Course Packet:" or a line that is not a citation.
 */
export function textCitation(line: string): TextCitation | null {
  const l = line.replace(/\s+/g, " ").trim();
  if (!l || /:$/.test(l) || l.length > 400) return null;
  const isbn = /ISBN(?:-1[03])?:?\s*([\d-]{10,17}X?)/i.exec(l)?.[1]?.replace(/-/g, "") ?? null;
  const edited = /\bEdited by\b|\beds?\.\s/i.test(l);
  // Last, First. Title. Publisher...
  const mla = /^(?:The readings are assigned in:\s*)?([A-Z][\p{L}'’-]+(?: [A-Z][\p{L}'’-]+)?), ([A-Z][\p{L}.'’ -]{0,40}?)\. (.+)$/u.exec(l);
  if (mla) {
    const [, last, first, rest] = mla;
    const author = `${first!.trim()} ${last}`;
    const chapter = /^["“]/.test(rest!) ? /\bIn ([^.]+?)\.(?:\s|$)/.exec(rest!)?.[1] : null;
    if (/^["“]/.test(rest!) && !chapter) return null;
    const title = (chapter ?? /^(.+?)\.(?:\s|$)/.exec(rest!)?.[1] ?? rest!).replace(/["“”]/g, "").trim();
    return title ? { title, author: chapter && edited ? null : author, isbn } : null;
  }
  // Author, Title
  const plain = /^([A-Z][\p{L}.'’ -]{1,40}?), (.+)$/u.exec(l);
  if (plain && plain[1]!.split(" ").length <= 4 && !/[.:;]/.test(plain[2]!.replace(/\.$/, ""))) return { title: plain[2]!.replace(/\.$/, "").trim(), author: plain[1]!.trim(), isbn };
  return isbn ? { title: "", author: null, isbn } : null;
}
