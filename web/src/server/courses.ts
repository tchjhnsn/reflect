// Courses taken in Reflect: each course's lectures and each lecture read from its publisher's public pages
// (@reflect/core/courses), cached a day by Next's fetch cache and never stored in the database.
import { courseInfo, courseLecture, courseLectures, textCitation, type CourseInfo, type CourseSite, type Lecture, type LectureRef } from "@reflect/core/courses";
import { searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { openLibrary } from "./browse";

const page = async (url: string): Promise<string | null> => {
  try {
    const r = await fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 }, signal: AbortSignal.timeout(8000) });
    return r.ok ? await r.text() : null;
  } catch {
    return null;
  }
};

export async function lecturesOf(url: string | null): Promise<{ site: CourseSite; url: string; lectures: LectureRef[] } | null> {
  return url ? courseLectures(url, page) : null;
}

export async function lectureOf(site: CourseSite, ref: LectureRef): Promise<Lecture | null> {
  return courseLecture(site, ref, page);
}

export async function courseInfoOf(url: string): Promise<CourseInfo | null> {
  return courseInfo(url, page);
}

export type SyllabusText = { line: string; book: CatalogEntry | null };

const words = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[^\p{L}\p{N} ]/gu, " ").split(/\s+/).filter((w) => w.length > 2 && !["the", "and", "from", "with"].includes(w));

/**
 * The books a syllabus asks for, each line found in Open Library (by its ISBN, else its title and author) so it shows
 * as a book; a line that is not a citation, or whose best match does not share the title's words, stays a line.
 */
export async function syllabusTexts(lines: readonly string[]): Promise<SyllabusText[]> {
  return Promise.all(lines.slice(0, 16).map(async (line) => {
    const c = textCitation(line);
    if (!c) return { line, book: null };
    try {
      const want = words(c.title);
      const hits = c.isbn ? await searchBooks(`isbn:${c.isbn}`, openLibrary, 1) : [];
      const pool = hits.length ? hits : await searchBooks([c.title, c.author].filter(Boolean).join(" "), openLibrary, 5);
      const book = pool.find((b) => { const got = new Set(words(b.title)); return want.length ? want.filter((w) => got.has(w)).length >= Math.min(2, want.length) : true; }) ?? null;
      return { line, book };
    } catch {
      return { line, book: null };
    }
  }));
}
