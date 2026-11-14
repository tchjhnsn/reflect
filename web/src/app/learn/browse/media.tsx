import Link from "next/link";
import type { ReactNode } from "react";
import { BookOpenIcon, FilmIcon, GraduationCapIcon, ImageIcon, MicIcon, MusicIcon, PresentationIcon, Gamepad2Icon, MessagesSquareIcon, TvIcon, DramaIcon } from "lucide-react";
import { browseWay, playsHere, type BrowseMedium } from "@reflect/core/browse-media";
import type { Found } from "@reflect/core/lookup";
import { WAYS, mediumOf, type Medium } from "@reflect/core/ways";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { browseAddAction } from "../actions";
import { LocalDay } from "../items/local-day";

const ICON: Record<Medium | "books", ReactNode> = {
  books: <BookOpenIcon />, course: <GraduationCapIcon />, lecture: <PresentationIcon />, debate: <MessagesSquareIcon />, discussion: <MessagesSquareIcon />,
  podcast: <MicIcon />, film: <FilmIcon />, series: <TvIcon />, theatre: <DramaIcon />, music: <MusicIcon />, art: <ImageIcon />, game: <Gamepad2Icon />, article: <BookOpenIcon />,
};
export const wayLabel = (m: BrowseMedium) => { const w = browseWay(m); return w === "leisure" ? "Rest and leisure" : WAYS.find((x) => x.id === w)!.label; };

export type Find = Found & { here?: boolean };
/** A find from a catalog or a list, keyed as the person's own items are (medium and title). */
export const keyOf = (f: { medium: string; title: string }) => `${f.medium}|${f.title.toLowerCase()}`;

/**
 * One find as a card: its picture, title and maker, its medium and way of learning, and Here when it can be done in
 * Reflect. Want to keeps it on your shelves; Start here keeps it and opens it to begin at once; a kept one opens.
 */
function MediaCard({ f, mine, showMedium }: { f: Find; mine: Map<string, string>; showMedium: boolean }) {
  const id = mine.get(keyOf(f));
  const here = f.here ?? playsHere(f);
  const m = mediumOf(f.medium);
  const square = f.medium === "music" || f.medium === "podcast" || f.medium === "art";
  return (
    <li className="flex min-w-0 flex-col gap-2">
      <div className={`bg-muted text-muted-foreground relative flex items-center justify-center overflow-hidden rounded-lg border [&_svg]:size-8 ${square ? "aspect-square" : f.medium === "course" || f.medium === "lecture" || f.medium === "debate" ? "aspect-video" : "aspect-[2/3]"}`}>
        {f.image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={f.image} alt="" loading="lazy" className="size-full object-cover" />
        ) : ICON[f.medium]}
        {here ? <span className="bg-background/90 text-foreground absolute top-1.5 left-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium" title="Can be done in Reflect">Here</span> : null}
      </div>
      <div className="min-w-0 space-y-0.5">
        <p className="line-clamp-2 text-sm font-medium">{id ? <Link href={`/learn/items/${id}`} className="hover:underline">{f.title}</Link> : f.title}</p>
        {f.by ? <p className="text-muted-foreground line-clamp-1 text-xs">{f.by}</p> : null}
        <p className="text-muted-foreground text-[11px]">{[showMedium ? m.label : null, wayLabel(f.medium), f.year].filter(Boolean).join(" · ")}</p>
      </div>
      <div className="mt-auto">
        {id ? <Button asChild size="sm" variant="outline" className="h-7 w-full"><Link href={`/learn/items/${id}`}>Open</Link></Button> : (
          <form action={browseAddAction} className="grid gap-1.5">
            <input type="hidden" name="medium" value={f.medium} /><input type="hidden" name="title" value={f.title} />
            {f.by ? <input type="hidden" name="by" value={f.by} /> : null}{f.url ? <input type="hidden" name="url" value={f.url} /> : null}{f.image ? <input type="hidden" name="image" value={f.image} /> : null}
            <LocalDay />
            {here ? <Button size="sm" className="h-7" name="go" value="start">Start here</Button> : null}
            <Button size="sm" variant="outline" className="h-7">Want to</Button>
          </form>
        )}
      </div>
    </li>
  );
}

export function MediaGrid({ finds, mine, showMedium = false, row = false }: { finds: Find[]; mine: Map<string, string>; showMedium?: boolean; row?: boolean }) {
  return (
    <ul className={row ? "-mx-4 flex snap-x gap-4 overflow-x-auto px-4 pb-2 sm:mx-0 sm:grid sm:grid-cols-4 sm:overflow-visible sm:px-0 lg:grid-cols-6 [&>li]:w-36 [&>li]:shrink-0 [&>li]:snap-start sm:[&>li]:w-auto" : "grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-6"}>
      {finds.map((f) => <MediaCard key={`${keyOf(f)}|${f.url ?? ""}`} f={f} mine={mine} showMedium={showMedium} />)}
    </ul>
  );
}

/** A titled row of finds, with a link to see the whole medium. */
export function MediaRow({ title, finds, mine, href, showMedium, note }: { title: ReactNode; finds: Find[] | null; mine: Map<string, string>; href?: string; showMedium?: boolean; note?: ReactNode }) {
  if (!finds?.length) return null;
  return (
    <section className="space-y-4">
      <div className="flex items-baseline justify-between gap-3 pt-2">
        <h2 className="text-lg font-semibold">{title}</h2>
        {href ? <Link href={href} className="text-muted-foreground shrink-0 text-sm underline-offset-4 hover:underline">See all</Link> : null}
      </div>
      {note ? <p className="text-muted-foreground -mt-2 text-sm">{note}</p> : null}
      <MediaGrid finds={finds} mine={mine} showMedium={showMedium} row />
    </section>
  );
}
