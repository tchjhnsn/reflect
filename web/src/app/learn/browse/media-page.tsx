import Link from "next/link";
import type { ReactNode } from "react";
import { BROWSE_MEDIA, browseWay, type BrowseMedium } from "@reflect/core/browse-media";
import { trendingBooks } from "@reflect/core/openlibrary";
import { WAYS, type Medium, type Way } from "@reflect/core/ways";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PAGE } from "@/components/page-header";
import { learnView } from "@/server/learn-store";
import { listItems } from "@/server/ways-store";
import { filmsToWatch, finds, listFinds } from "@/server/browse-media";
import { openLibrary } from "@/server/browse";
import { LearnHeader } from "../header";
import { BookGrid } from "../book-card";
import { Menu } from "./category-menu";
import { MediaGrid, MediaRow, keyOf, type Find } from "./media";

export type WayPick = Way | "leisure";
export const WAY_PICKS: readonly { id: WayPick; label: string }[] = [...WAYS.map((w) => ({ id: w.id, label: w.label })), { id: "leisure", label: "Rest and leisure" }];
export const isWayPick = (v: string): v is WayPick => WAY_PICKS.some((w) => w.id === v);

/** Books belong to teaching, imitation (stories) and dialectic (dialogues); every other medium to its own way. */
const inWay = (m: BrowseMedium, way: WayPick | null) => !way || (m === "books" ? way === "teaching" || way === "imitation" || way === "dialectic" : browseWay(m) === way);
const href = (m: BrowseMedium | "all", way: WayPick | null, q?: string) => {
  const p = new URLSearchParams();
  if (m !== "all") p.set("m", m);
  if (way) p.set("way", way);
  if (q) p.set("q", q);
  const s = p.toString();
  return `/learn/browse${s ? `?${s}` : ""}`;
};

/**
 * Browse's top: every medium as a tab (books, courses, lectures, debates, podcasts, plays, films, series, music, art,
 * games, or all of them), and a way of learning to narrow any list to (Torian, 2026-11-14: "Both").
 */
export function BrowseNav({ m, way, children }: { m: BrowseMedium | "all"; way: WayPick | null; children?: ReactNode }) {
  const media = BROWSE_MEDIA.filter((x) => inWay(x.id, way));
  return (
    <div className="space-y-3">
      <nav aria-label="What to browse" className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
        <ul className="flex gap-1.5 pb-1">
          {[{ id: "all" as const, label: "All" }, ...media].map((x) => (
            <li key={x.id}>
              <Link href={href(x.id, way)} aria-current={x.id === m ? "page" : undefined} className={`inline-flex h-8 items-center rounded-full border px-3 text-sm whitespace-nowrap transition ${x.id === m ? "bg-foreground text-background border-foreground" : "hover:bg-muted"}`}>{x.label}</Link>
            </li>
          ))}
        </ul>
      </nav>
      <div className="flex flex-wrap items-center gap-2">
        {children}
        <Menu
          label={way ? WAY_PICKS.find((w) => w.id === way)!.label : "Every way of learning"}
          lead={1}
          items={[{ href: href(m, null), label: "Every way of learning", on: !way }, ...WAY_PICKS.map((w) => ({ href: href(inWay(m === "all" ? "books" : m, w.id) || m === "all" ? m : "all", w.id), label: w.label, on: way === w.id }))]}
        />
      </div>
    </div>
  );
}

const SEARCH_HINT: Record<string, string> = {
  all: "Search books, courses, podcasts, films and art", course: "Search courses from Open Yale and MIT", lecture: "Search lectures", debate: "Search debates",
  podcast: "Search podcasts", theatre: "Search plays", film: "Search films", series: "Search series", music: "Search music", art: "Search the Art Institute of Chicago", game: "Search games",
};
const NOT_HERE: Partial<Record<Medium, string>> = {
  series: "Series cannot be watched in Reflect yet; keep them here and say what they taught you.",
  music: "Music cannot be heard in Reflect yet, apart from recordings in the public domain.",
  game: "Games cannot be played in Reflect yet.",
  film: "Films marked Here play in Reflect from the Internet Archive's public domain; other films cannot be watched here yet.",
};

/** Browse for every medium but books: all of them together, or one medium, searched or from its catalogs and lists. */
export async function MediaBrowse({ userId, m, way, q }: { userId: string; m: Medium | "all"; way: WayPick | null; q: string }) {
  const [view, items] = await Promise.all([learnView(userId), listItems(userId)]);
  const mine = new Map(items.map((i) => [keyOf(i), i.id]));
  const subjects = [...new Set([...view.books.flatMap((b) => b.subjects), ...view.known, ...view.shown].map((s) => s.name))].slice(0, 2);
  const show = (x: Medium) => inWay(x, way);
  const here = (fs: Find[]) => fs.map((f) => ({ ...f, here: true }));

  let body: ReactNode;
  if (m === "all") {
    if (q) {
      const media: Medium[] = (["course", "podcast", "film", "theatre", "art", "music", "series", "game"] as const).filter(show);
      const results = await Promise.all(media.map((x) => finds(x, q).then((r) => r.found.slice(0, 6)).catch(() => [])));
      body = (
        <div className="space-y-8">
          {inWay("books", way) ? <p className="text-sm"><Link href={`/learn/browse?q=${encodeURIComponent(q)}`} className="underline underline-offset-4">Books matching “{q}”</Link></p> : null}
          {media.map((x, i) => <MediaRow key={x} title={BROWSE_MEDIA.find((b) => b.id === x)!.label} finds={results[i]!} mine={mine} href={href(x, way, q)} />)}
          {results.every((r) => !r.length) ? <p className="text-muted-foreground text-sm">Nothing found beyond books.</p> : null}
        </div>
      );
    } else {
      const [courses, books, podcasts, films, art] = await Promise.all([
        show("course") ? finds("course").then((r) => r.found.slice(0, 12)) : null,
        inWay("books", way) ? trendingBooks(openLibrary, 12).catch(() => null) : null,
        show("podcast") ? finds("podcast", subjects[0] ?? "philosophy").then((r) => r.found.slice(0, 12)) : null,
        show("film") ? filmsToWatch().then((f) => f.slice(0, 12)) : null,
        show("art") ? finds("art").then((r) => r.found.slice(0, 12)) : null,
      ]);
      body = (
        <div className="space-y-8">
          <MediaRow title="Courses to take here" finds={courses} mine={mine} href={href("course", way)} />
          {books?.length ? (
            <section className="space-y-4">
              <div className="flex items-baseline justify-between gap-3 pt-2"><h2 className="text-lg font-semibold">Books read this week</h2><Link href={href("books", way)} className="text-muted-foreground text-sm underline-offset-4 hover:underline">See all</Link></div>
              <BookGrid books={books} mine={new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]))} row />
            </section>
          ) : null}
          <MediaRow title={subjects[0] ? <>Podcasts on {subjects[0]}</> : "Podcasts on philosophy"} finds={podcasts} mine={mine} href={href("podcast", way)} />
          {show("lecture") ? <MediaRow title="Lectures" finds={listFinds("lecture")} mine={mine} href={href("lecture", way)} /> : null}
          {show("debate") ? <MediaRow title="Debates" finds={listFinds("debate")} mine={mine} href={href("debate", way)} /> : null}
          {show("theatre") ? <MediaRow title="Plays to read here" finds={here(listFinds("theatre"))} mine={mine} href={href("theatre", way)} /> : null}
          <MediaRow title="Films to watch here" finds={films} mine={mine} href={href("film", way)} note="Feature films in the public domain, from the Internet Archive." />
          <MediaRow title="Art to look at here" finds={art} mine={mine} href={href("art", way)} note="Public-domain works at the Art Institute of Chicago." />
        </div>
      );
    }
  } else {
    const lists = listFinds(m);
    const r = q || m === "course" || m === "art" ? await finds(m, q) : { found: [] as Find[] };
    const extra: ReactNode[] = [];
    if (!q && m === "film") extra.push(<MediaRow key="free" title="Free to watch here" finds={await filmsToWatch()} mine={mine} note="Feature films in the public domain, from the Internet Archive." />);
    if (!q && m === "podcast") {
      const topics = [...subjects, "philosophy", "history", "theology"].filter((t, i, a) => a.indexOf(t) === i).slice(0, 4);
      const rows = await Promise.all(topics.map((t) => finds("podcast", t).then((x) => x.found.slice(0, 12)).catch(() => [])));
      topics.forEach((t, i) => extra.push(<MediaRow key={t} title={<>On {t}</>} finds={rows[i]!} mine={mine} href={href("podcast", way, t)} />));
    }
    body = (
      <div className="space-y-8">
        {NOT_HERE[m] ? <p className="text-muted-foreground text-sm">{NOT_HERE[m]}</p> : null}
        {r.error ? <p className="text-muted-foreground text-sm">{r.error}</p> : null}
        {r.found.length ? (
          <section className="space-y-4">
            {q ? <h2 className="pt-2 text-lg font-semibold">Results for “{q}”</h2> : <h2 className="pt-2 text-lg font-semibold">{m === "course" ? "Courses to take here, from Open Yale and MIT" : "The Art Institute of Chicago's best-known works"}</h2>}
            <MediaGrid finds={r.found} mine={mine} />
          </section>
        ) : q && !r.error ? <p className="text-muted-foreground text-sm">Nothing found.</p> : null}
        {extra}
        {lists.length && !q ? <MediaRow title="From the lists" finds={m === "theatre" ? here(lists) : lists} mine={mine} note={<Link href="/learn/lists" className="underline underline-offset-4">Lists beyond books</Link>} /> : null}
        {!q && (m === "lecture" || m === "debate") ? <p className="text-muted-foreground text-sm">Any recorded {m} on YouTube, Vimeo or the Internet Archive plays here: paste its link under My shelves.</p> : null}
      </div>
    );
  }

  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-5">
        <BrowseNav m={m} way={way}>
          <form action="/learn/browse" className="flex min-w-0 flex-1 basis-72 gap-2">
            {m !== "all" ? <input type="hidden" name="m" value={m} /> : <input type="hidden" name="m" value="all" />}
            {way ? <input type="hidden" name="way" value={way} /> : null}
            <label htmlFor="q" className="sr-only">{SEARCH_HINT[m]}</label>
            <Input id="q" name="q" defaultValue={q} placeholder={SEARCH_HINT[m]} className="h-10" />
            <Button type="submit" className="h-10">Search</Button>
          </form>
        </BrowseNav>
        {body}
        <p className="text-muted-foreground pt-6 text-center text-xs">Courses from Open Yale Courses and MIT OpenCourseWare (through MIT Learn), podcasts from Apple&apos;s podcast search, films from Wikidata and the Internet Archive, art from the Art Institute of Chicago, music from MusicBrainz, series from TVmaze, games from Steam.</p>
      </div>
    </main>
  );
}
