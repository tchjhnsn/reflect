// Courses taken in Reflect: each course's lectures and each lecture read from its publisher's public pages
// (@reflect/core/courses), cached a day by Next's fetch cache and never stored in the database.
import { courseInfo, courseLecture, courseLectures, type CourseInfo, type CourseSite, type Lecture, type LectureRef } from "@reflect/core/courses";

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
