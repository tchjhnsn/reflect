import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { STANCES } from "@reflect/core/claims";
import { embedOf } from "@reflect/core/media";
import { WAYS, mediumOf, placeOf } from "@reflect/core/ways";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { getItem, listUnits } from "@/server/ways-store";
import { lecturesOf } from "@/server/courses";
import { reasonFor } from "@/server/learn-reason-store";
import { showFor } from "@/server/podcasts";
import { openLibrary } from "@/server/browse";
import "@/server/free";
import { searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { clock } from "@reflect/core/media";
import { COURSE_SITES } from "@reflect/core/courses";
import { listIdeas, type IdeaOut } from "@/server/ideas-store";
import { ideasNaming } from "@reflect/core/ideas";
import { learnView } from "@/server/learn-store";
import { listClaims } from "@/server/claims-store";
import { publishedBy } from "@/server/published-store";
import { LearnHeader } from "../../header";
import {
  addItemSubjectsAction, addSessionAction, changeItemAction, editItemAction, keepItemClaimAction, removeItemAction, removeItemSubjectAction, removeSessionAction,
  setItemDatesAction, setItemFactsAction, setItemStanceAction, setTranscriptAction, unpublishQuoteFormAction,
} from "../../actions";
import { LocalDay } from "../local-day";
import { ItemMedia } from "./media";
import { PublishLesson } from "./publish-lesson";
import { Artwork } from "./artwork";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

const SELECT = "border-input bg-background h-9 w-full rounded-md border px-3 text-sm";
const STANCE_LABEL: Record<string, string> = { belief: "Belief", disbelief: "Disbelief", suspension: "Suspension of judgment" };
const Note = ({ children }: { children: ReactNode }) => <p className="text-muted-foreground text-sm">{children}</p>;
const section = (title: string, body: ReactNode, id?: string) => (
  <Card className="gap-3 py-5" id={id}>
    <CardHeader className="px-5"><CardTitle className="text-sm">{title}</CardTitle></CardHeader>
    <CardContent className="space-y-3 px-5">{body}</CardContent>
  </Card>
);
const fmt = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });

/**
 * One learn item: its status and dates, its way, what it taught you (dated lessons, the latest publishable), and
 * what its way asks for: the claims argued in a debate and whether you took part, the parts of a course, the
 * sessions of a practice. A video or recording plays here with your place kept and notes at a moment; a pasted
 * transcript is searched and quoted. Subjects, the book it adapts, editing and removing are here too.
 */
export default async function ItemPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const item = await getItem(user.id, (await params).id);
  if (!item) notFound();
  const [view, claims, published, ideas] = await Promise.all([learnView(user.id), listClaims(user.id).catch(() => []), publishedBy(user.id).catch(() => []), listIdeas(user.id).catch((): IdeaOut[] => [])]);
  // An item merged from an older input keeps its id, so an idea may name it either way.
  const bearing = [...new Map([...ideasNaming(ideas, `item:${item.id}`, item.title), ...ideasNaming(ideas, `input:${item.id}`, item.title)].map((b) => [b.idea.id, b])).values()];
  const [course, show, units, reason] = await Promise.all([
    item.medium === "course" ? lecturesOf(item.url) : null,
    item.medium === "podcast" ? showFor(item.url) : null,
    item.medium === "course" || item.medium === "podcast" ? listUnits(user.id, item.id) : [],
    item.medium === "course" && item.url ? reasonFor(user.id, item.url) : null,
  ]);
  // A play or a debate kept in writing is read (and often heard) in Reflect, as a book.
  const text = (item.medium === "theatre" || (item.medium === "debate" && !embedOf(item.url)))
    ? (await searchBooks([item.title.replace(/\s*\(.*\)\s*$/, ""), item.by].filter(Boolean).join(" "), openLibrary, 5).catch((): CatalogEntry[] => [])).find((b) => b.free)
    : undefined;
  const lectures = course?.lectures ?? [];
  const finished = new Set(units.filter((u) => u.doneOn).map((u) => u.unit));
  const begun = new Map(units.map((u) => [u.unit, u]));
  const nextLecture = lectures.find((l) => !finished.has(String(l.n)));
  const m = mediumOf(item.medium);
  const place = placeOf(item);
  const way = item.way ?? m.way;
  const embed = embedOf(item.url);
  const argued = claims.filter((c) => item.claimIds.includes(c.id));
  const books = view.books.filter((b) => b.catalog?.work);
  const adapts = item.work ? books.find((b) => b.catalog!.work === item.work) : undefined;
  const subjectIds = new Set(item.subjects.map((s) => s.id));
  const related = view.books.filter((b) => b.subjects.some((s) => subjectIds.has(s.id)));
  const lessonsOut = published.filter((p) => p.kind === "lesson" && p.title === item.title);
  const practice = way === "practice" || item.medium === "game";
  const since = new Date(Date.now() - 30 * 86400 * 1000).toISOString().slice(0, 10);
  const recent = item.sessions.filter((s) => s.day >= since);
  const minutes = item.sessions.reduce((n, s) => n + (s.minutes ?? 0), 0);
  const hidden = <input type="hidden" name="item" value={item.id} />;
  return (
    <main className={PAGE}>
      <LearnHeader on="shelves" />
      <div className="mt-6 space-y-6">
        <p className="text-muted-foreground text-sm"><Link href={`/learn#${place}`} className="hover:underline">My shelves</Link> · {place === "leisure" ? "Rest and leisure" : WAYS.find((w) => w.id === place)!.label}</p>
        <header className="flex gap-4">
          {item.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={item.image} alt="" className="h-28 w-28 shrink-0 rounded-lg object-cover" />
          ) : null}
          <div className="min-w-0 space-y-1">
            <div className="flex flex-wrap items-center gap-2"><Badge variant="secondary">{m.label}</Badge>{item.tookPart ? <Badge variant="outline">Took part</Badge> : null}</div>
            <h1 className="text-2xl font-semibold tracking-tight">{item.title}</h1>
            {item.by ? <p className="text-muted-foreground">{item.by}</p> : null}
            {course ? <p className="text-muted-foreground text-sm">From {COURSE_SITES[course.site].name} · {lectures.length} lectures · {finished.size} finished · <Link href={`/learn/courses?u=${encodeURIComponent(course.url)}&t=${encodeURIComponent(item.title)}`} className="underline underline-offset-4">About the course and its syllabus</Link></p> : null}
            {course ? <p className="max-w-prose text-sm leading-relaxed"><span className="font-medium">Why I am learning this: </span>{reason ? <>{reason.cause.length > 220 ? `${reason.cause.slice(0, 217)}...` : reason.cause} <Link href={`/journal/w/${reason.workId}`} className="text-muted-foreground underline underline-offset-4">In your journal</Link></> : <Link href={`/learn/courses/why?u=${encodeURIComponent(course.url)}&t=${encodeURIComponent(item.title)}`} className="underline underline-offset-4">Write it</Link>}</p> : null}
            {adapts ? <p className="text-sm">Adapts <Link href={`/learn/works/${item.work}`} className="underline underline-offset-4">{adapts.title}</Link></p> : null}
          </div>
        </header>

        <form action={changeItemAction} className="flex flex-wrap gap-2">
          {hidden}<LocalDay />
          {(["want", "doing", "done"] as const).map((s) => (
            <Button key={s} size="sm" name="status" value={s} variant={item.status === s ? "default" : "outline"} aria-pressed={item.status === s}>{s === "want" ? "Want to" : s === "doing" ? m.doing : m.done}</Button>
          ))}
        </form>

        <form action={setItemDatesAction} className="flex flex-wrap items-end gap-3 text-sm">
          {hidden}
          <label className="space-y-1">Started<Input type="date" name="started" defaultValue={item.startedOn ?? ""} className="h-9 w-40" /></label>
          <label className="space-y-1">Finished<Input type="date" name="finished" defaultValue={item.finishedOn ?? ""} className="h-9 w-40" /></label>
          <Button size="sm" variant="outline">Save dates</Button>
        </form>

        {item.medium === "art" && item.image ? section("Look at it here", <Artwork src={item.image} title={item.title} />, "artwork") : null}

        {text ? section(text.free?.read ? "Read it here" : "Listen here", (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {text.free?.read ? <Button size="sm" asChild><Link href={`/learn/works/${text.work}/read`}>Read {text.title}</Link></Button> : null}
            {text.free?.listen ? <Button size="sm" variant="outline" asChild><Link href={`/learn/works/${text.work}#listen`}>Listen to it</Link></Button> : null}
            <span className="text-muted-foreground">{text.authors.join(", ")}{text.year ? `, ${text.year}` : ""}, in Reflect&apos;s reader with highlights and excerpts.</span>
          </div>
        ), "text") : null}

        {lectures.length ? section("Lectures", (
          <>
            {nextLecture ? <Button size="sm" asChild><Link href={`/learn/items/${item.id}/lectures/${nextLecture.n}`}>{finished.size || begun.size ? "Continue" : "Start"}: {nextLecture.title}</Link></Button> : <Note>You have finished every lecture.</Note>}
            <ol className="divide-y text-sm">
              {lectures.map((l) => {
                const u = begun.get(String(l.n));
                return (
                  <li key={l.n} className="flex items-center gap-3 py-2">
                    <span className="text-muted-foreground w-6 shrink-0 text-right tabular-nums">{l.n}</span>
                    <Link href={`/learn/items/${item.id}/lectures/${l.n}`} className="min-w-0 flex-1 underline-offset-4 hover:underline">{l.title}</Link>
                    <span className="text-muted-foreground shrink-0 text-xs">{u?.doneOn ? "Finished" : u?.place ? "Under way" : ""}</span>
                  </li>
                );
              })}
            </ol>
            <p className="text-muted-foreground text-xs">Each lecture plays here from {COURSE_SITES[course!.site].name}, under the {COURSE_SITES[course!.site].license} license, with its transcript beside it.</p>
          </>
        ), "lectures") : null}

        {show?.episodes.length ? section("Episodes", (
          <>
            <ol className="divide-y text-sm">
              {show.episodes.slice(0, 50).map((e) => {
                const u = begun.get(e.key);
                return (
                  <li key={e.key} className="flex items-center gap-3 py-2">
                    <Link href={`/learn/items/${item.id}/episodes/${e.key}`} className="min-w-0 flex-1 underline-offset-4 hover:underline">{e.title}</Link>
                    <span className="text-muted-foreground shrink-0 text-xs tabular-nums">{u?.doneOn ? "Heard" : u?.place ? "Under way" : [e.date, e.seconds ? clock(e.seconds) : null].filter(Boolean).join(" · ")}</span>
                  </li>
                );
              })}
            </ol>
            <p className="text-muted-foreground text-xs">Each episode plays here from the show&apos;s own feed{show.episodes.length > 50 ? `; the latest 50 of ${show.episodes.length}` : ""}.</p>
          </>
        ), "episodes") : item.medium === "podcast" && item.url ? section("Episodes", <Note>The show&apos;s feed could not be read just now.</Note>) : null}

        {embed || item.notes.some((n) => !n.unit) || item.transcript ? section(embed ? `${m.done === "Heard" ? "Listen" : "Watch"} here` : "Notes", <ItemMedia id={item.id} title={item.title} by={item.by} embed={embed} place={item.place} notes={item.notes.filter((n) => !n.unit)} transcript={item.transcriptText} />) : null}

        {section("What it taught you", (
          <>
            {item.lessons.length ? (
              <ol className="space-y-2">
                {item.lessons.map((l) => <li key={l.id} className="rounded-md border p-3 text-sm"><p className="whitespace-pre-wrap">{l.text}</p><p className="text-muted-foreground mt-1 text-xs">{fmt(l.at.slice(0, 10))}</p></li>)}
              </ol>
            ) : <Note>{m.lesson ? "Until you say what it taught you, it is kept under Rest and leisure." : "Nothing written yet."}</Note>}
            <form action={changeItemAction} className="space-y-2">
              {hidden}
              <Textarea name="lesson" rows={3} maxLength={4000} aria-label={item.lessons.length ? "Add a later lesson" : "What it taught you"} placeholder={item.lessons.length ? "Something it taught you since" : "What it taught you"} required />
              <Button size="sm">Keep</Button>
            </form>
            {item.lesson ? <PublishLesson id={item.id} /> : null}
            {lessonsOut.map((p) => (
              <form key={p.id} action={unpublishQuoteFormAction} className="flex flex-wrap items-center gap-2 text-sm">
                <input type="hidden" name="id" value={p.id} />{hidden}
                <span>Published at <a href={`/q/${p.id}`} className="underline underline-offset-4">/q/{p.id}</a></span>
                <Button size="sm" variant="ghost">Take down</Button>
              </form>
            ))}
          </>
        ))}

        {bearing.length ? section("Your ideas", (
          <div className="space-y-3">
            {bearing.map(({ idea, how }) => (
              <div key={idea.id} className="text-sm">
                <p className="font-medium">{idea.title}</p>
                {idea.statement ? <p className="text-muted-foreground line-clamp-3">{idea.statement}</p> : null}
                <p className="text-muted-foreground text-xs">{how === "from" ? `Came from this ${m.label.toLowerCase()}` : how === "maybe" ? `May have come from this ${m.label.toLowerCase()}` : `This ${m.label.toLowerCase()} ${how} it`}</p>
              </div>
            ))}
          </div>
        )) : null}

        {!lectures.length && (item.medium === "course" || item.parts) ? section("Parts", (
          <form action={setItemFactsAction} className="flex flex-wrap items-center gap-2 text-sm">
            {hidden}
            <Button size="sm" variant="outline" name="step" value="-1" aria-label="One part back">−</Button>
            <span className="tabular-nums">{item.part ?? 0} of {item.parts ?? "?"}</span>
            <Button size="sm" variant="outline" name="step" value="1" aria-label="One more part">+</Button>
            <label className="ml-4 flex items-center gap-2">Parts in all<Input type="number" name="parts" min={1} max={10000} defaultValue={item.parts ?? ""} className="h-8 w-24" /></label>
            <Button size="sm" variant="ghost">Save</Button>
          </form>
        )) : null}

        {way === "dialectic" ? section("The claims argued", (
          <>
            <form action={setItemFactsAction} className="flex flex-wrap items-center gap-2 text-sm">
              {hidden}
              <span>You</span>
              <Button size="sm" name="took_part" value="yes" variant={item.tookPart ? "default" : "outline"} aria-pressed={item.tookPart}>took part</Button>
              <Button size="sm" name="took_part" value="no" variant={!item.tookPart ? "default" : "outline"} aria-pressed={!item.tookPart}>only heard it</Button>
            </form>
            {argued.length ? argued.map((c) => (
              <div key={c.id} className="space-y-2 rounded-md border p-3">
                <p className="text-sm font-medium">{c.text}</p>
                <p className="text-muted-foreground text-xs">Argued by {c.holders.map((h) => h.name).join("; ") || "no one named"}{c.stance ? ` · your stance: ${STANCE_LABEL[c.stance.stance]?.toLowerCase()}` : ""}</p>
                <form action={setItemStanceAction} className="flex flex-wrap gap-2">
                  {hidden}<input type="hidden" name="claim" value={c.id} />
                  {STANCES.map((s) => <Button key={s.id} size="sm" name="stance" value={s.id} variant={c.stance?.stance === s.id ? "default" : "outline"}>{STANCE_LABEL[s.id]}</Button>)}
                </form>
              </div>
            )) : <Note>Keep each claim argued, who argued it, and where you stand. Your stance can change; earlier ones are kept.</Note>}
            <form action={keepItemClaimAction} className="space-y-3">
              {hidden}
              <label className="block space-y-1.5 text-sm font-medium">Claim<Input name="claim" required maxLength={500} /></label>
              <label className="block space-y-1.5 text-sm font-medium">Who argued it, one per line<Textarea name="holders" rows={2} defaultValue={item.by?.split(/ and |, /).length === 1 ? item.by ?? "" : ""} /></label>
              <div className="flex flex-wrap gap-2">{STANCES.map((s) => <Button key={s.id} size="sm" variant="outline" name="stance" value={s.id}>Keep as {STANCE_LABEL[s.id]!.toLowerCase()}</Button>)}</div>
            </form>
          </>
        )) : null}

        {practice ? section("Practice", (
          <>
            <Note>{item.sessions.length ? `${recent.length} ${recent.length === 1 ? "session" : "sessions"} in the last 30 days · ${item.sessions.length} in all${minutes ? ` · ${Math.round(minutes / 6) / 10} hours` : ""}` : "We become what we repeatedly do. Keep each time you practise."}</Note>
            <form action={addSessionAction} className="flex flex-wrap items-end gap-2 text-sm">
              {hidden}<LocalDay />
              <label className="space-y-1">Day (today if blank)<Input type="date" name="on" className="h-9 w-40" /></label>
              <label className="space-y-1">Minutes<Input type="number" name="minutes" min={1} max={1440} className="h-9 w-24" /></label>
              <label className="min-w-40 flex-1 space-y-1">Note<Input name="note" maxLength={2000} className="h-9" /></label>
              <Button size="sm">Keep session</Button>
            </form>
            {item.sessions.length ? (
              <ul className="divide-y text-sm">
                {item.sessions.slice(0, 30).map((s) => (
                  <li key={s.id} className="flex items-center justify-between gap-2 py-2">
                    <span>{fmt(s.day)}{s.minutes ? ` · ${s.minutes} min` : ""}{s.note ? <span className="text-muted-foreground"> · {s.note}</span> : null}</span>
                    <form action={removeSessionAction}>{hidden}<input type="hidden" name="session" value={s.id} /><Button size="sm" variant="ghost">Remove</Button></form>
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        )) : null}

        {section("Way of learning", (
          <form action={changeItemAction} className="space-y-2">
            {hidden}
            <div className="flex flex-wrap gap-2">{WAYS.map((w) => <Button key={w.id} size="sm" name="way" value={w.id} variant={way === w.id ? "default" : "outline"} aria-pressed={way === w.id}>{w.label}</Button>)}</div>
            {item.way ? <Button size="sm" variant="ghost" name="way" value="">Back to {WAYS.find((w) => w.id === m.way)!.label.toLowerCase()}, from its medium</Button> : null}
          </form>
        ))}

        {section("Subjects", (
          <>
            {item.subjects.length ? (
              <div className="flex flex-wrap gap-1.5">
                {item.subjects.map((s) => (
                  <form key={s.id} action={removeItemSubjectAction}>{hidden}<input type="hidden" name="subject" value={s.id} /><Badge variant="secondary" className="gap-1">{s.name}<button aria-label={`Remove ${s.name}`} className="ml-1">×</button></Badge></form>
                ))}
              </div>
            ) : <Note>Subjects it is about, like a book&apos;s, count toward what you are learning.</Note>}
            <form action={addItemSubjectsAction} className="flex gap-2">{hidden}<Input name="subjects" placeholder="Justice, Utilitarianism" className="h-9" required /><Button size="sm" variant="outline">Add</Button></form>
            {related.length ? <p className="text-sm">Books on your shelves on the same subjects: {related.slice(0, 6).map((b, i) => <span key={b.id}>{i ? ", " : ""}{b.catalog?.work ? <Link href={`/learn/works/${b.catalog.work}`} className="underline underline-offset-4">{b.title}</Link> : b.title}</span>)}</p> : null}
          </>
        ))}

        {embed || item.transcript ? section("Transcript", (
          <form action={setTranscriptAction} className="space-y-2">
            {hidden}
            <Note>Paste the transcript (on YouTube: More, then Show transcript, then copy it) to search it and quote it with its moment. YouTube does not let other sites fetch it.</Note>
            <Textarea name="transcript" rows={4} defaultValue={item.transcriptText ?? ""} />
            <Button size="sm" variant="outline">{item.transcript ? "Save transcript" : "Keep transcript"}</Button>
          </form>
        )) : null}

        <details className="rounded-lg border p-4">
          <summary className="cursor-pointer text-sm font-medium">Edit</summary>
          <form action={editItemAction} className="mt-4 grid gap-3 sm:grid-cols-2">
            {hidden}
            <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">Title<Input name="title" defaultValue={item.title} required maxLength={300} /></label>
            <label className="block space-y-1.5 text-sm font-medium">By<Input name="by" defaultValue={item.by ?? ""} maxLength={200} /></label>
            <label className="block space-y-1.5 text-sm font-medium">Link<Input name="url" type="url" defaultValue={item.url ?? ""} maxLength={1000} /></label>
            <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">Picture (https link)<Input name="image" type="url" defaultValue={item.image ?? ""} maxLength={1000} /></label>
            {books.length ? (
              <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">Adapts or stages a book on your shelves
                <select name="work" className={SELECT} defaultValue={item.work ?? ""}>
                  <option value="">None</option>
                  {books.map((b) => <option key={b.id} value={b.catalog!.work}>{b.title}</option>)}
                </select>
              </label>
            ) : null}
            <div className="sm:col-span-2"><Button size="sm">Save</Button></div>
          </form>
        </details>

        <form action={removeItemAction}>{hidden}<Button size="sm" variant="ghost">Remove from your shelves</Button></form>
      </div>
    </main>
  );
}
