import Link from "next/link";
import type { ReactNode } from "react";
import { CheckIcon, ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import { LearnHeader } from "../header";

export type ShellLecture = { n: number; title: string; href: string; done: boolean };
export type ShellLink = { title: string; href: string };

/** Where a course's lecture opens: the kept course's own lecture page, else the lecture watched before keeping it. */
export const lectureHref = (courseUrl: string, n: number, title: string, keptId: string | undefined) =>
  keptId ? `/learn/items/${keptId}/lectures/${n}` : `/learn/courses/lecture?u=${encodeURIComponent(courseUrl)}&n=${n}${title ? `&t=${encodeURIComponent(title)}` : ""}`;
export const courseHref = (courseUrl: string, title: string) => `/learn/courses?u=${encodeURIComponent(courseUrl)}${title ? `&t=${encodeURIComponent(title)}` : ""}`;

/** A docs page's width: the Learn header above, then the course's sidebar, the page and what is on it. */
export const DOCS = "mx-auto w-full max-w-7xl px-4 py-6 sm:px-6";

function Nav({ course, overview, lectures, current }: { course: string; overview: string; lectures: ShellLecture[]; current: number | null }) {
  return (
    <nav aria-label="This course" className="space-y-4 text-sm">
      <Link href={overview} className={`block rounded-md px-2 py-1.5 font-medium ${current === null ? "bg-muted" : "hover:bg-muted/60"}`}>{course}</Link>
      {lectures.length ? (
        <div className="space-y-1">
          <p className="text-muted-foreground px-2 text-xs font-medium tracking-wide uppercase">Lectures</p>
          <ol className="space-y-px">
            {lectures.map((l) => (
              <li key={l.n}>
                <Link href={l.href} aria-current={current === l.n ? "page" : undefined} className={`flex gap-2 rounded-md px-2 py-1.5 leading-snug ${current === l.n ? "bg-muted text-foreground font-medium" : "text-muted-foreground hover:text-foreground hover:bg-muted/60"}`}>
                  <span className="w-5 shrink-0 text-right tabular-nums">{l.done ? <CheckIcon className="ml-auto size-4" aria-label="Finished" /> : l.n}</span>
                  <span className="min-w-0">{l.title}</span>
                </Link>
              </li>
            ))}
          </ol>
        </div>
      ) : null}
    </nav>
  );
}

/**
 * A course laid out as a documentation site is (after the Next.js docs): the course's lectures in a sidebar that stays
 * in view, the page in the middle, and what is on the page at the right; on a phone the sidebar folds into a menu at
 * the top. Previous and next close every lecture.
 */
export function CourseShell({ course, overview, lectures, current, toc, prev, next, on, sidebar = true, children }: {
  course: string; overview: string; lectures: ShellLecture[]; current: number | null;
  toc?: { id: string; label: string }[]; prev?: ShellLink | null; next?: ShellLink | null; on: "browse" | "shelves"; /** False on the course's own page, which lists the lectures itself. */ sidebar?: boolean; children: ReactNode;
}) {
  return (
    <main className={DOCS}>
      <LearnHeader on={on} />
      <div className={`mt-6 grid gap-8 ${sidebar ? `lg:grid-cols-[15rem_minmax(0,1fr)] ${toc?.length ? "xl:grid-cols-[15rem_minmax(0,1fr)_12rem]" : ""}` : toc?.length ? "lg:grid-cols-[minmax(0,1fr)_12rem]" : ""}`}>
        {sidebar ? <aside className="hidden lg:block">
          <div className="sticky top-6 -ml-2 max-h-[calc(100vh-3rem)] overflow-y-auto pr-2 pb-6"><Nav course={course} overview={overview} lectures={lectures} current={current} /></div>
        </aside> : null}
        <div className="min-w-0 space-y-8">
          {sidebar ? <details className="group rounded-lg border lg:hidden">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 text-sm font-medium [&::-webkit-details-marker]:hidden">
              <span className="truncate">{current === null ? course : `Lecture ${current} · ${course}`}</span>
              <ChevronDownIcon className="size-4 shrink-0 transition group-open:rotate-180" />
            </summary>
            <div className="max-h-[60vh] overflow-y-auto border-t p-2"><Nav course={course} overview={overview} lectures={lectures} current={current} /></div>
          </details> : null}
          {children}
          {prev || next ? (
            <nav aria-label="Previous and next" className="grid gap-3 border-t pt-6 sm:grid-cols-2">
              {prev ? (
                <Link href={prev.href} className="hover:bg-muted/50 rounded-lg border p-3 text-sm">
                  <span className="text-muted-foreground flex items-center gap-1 text-xs"><ChevronLeftIcon className="size-3.5" /> Previous</span>
                  <span className="mt-1 line-clamp-2 block font-medium">{prev.title}</span>
                </Link>
              ) : <span className="hidden sm:block" />}
              {next ? (
                <Link href={next.href} className="hover:bg-muted/50 rounded-lg border p-3 text-right text-sm">
                  <span className="text-muted-foreground flex items-center justify-end gap-1 text-xs">Next <ChevronRightIcon className="size-3.5" /></span>
                  <span className="mt-1 line-clamp-2 block font-medium">{next.title}</span>
                </Link>
              ) : null}
            </nav>
          ) : null}
        </div>
        {toc?.length ? (
          <aside className={sidebar ? "hidden xl:block" : "hidden lg:block"}>
            <div className="sticky top-6 space-y-2 text-sm">
              <p className="font-medium">On this page</p>
              <ul className="space-y-1.5">
                {toc.map((t) => <li key={t.id}><a href={`#${t.id}`} className="text-muted-foreground hover:text-foreground">{t.label}</a></li>)}
              </ul>
            </div>
          </aside>
        ) : null}
      </div>
    </main>
  );
}
