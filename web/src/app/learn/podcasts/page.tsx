import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { clock } from "@reflect/core/media";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { listItems, listUnits } from "@/server/ways-store";
import { showFor } from "@/server/podcasts";
import { ThingHeader } from "@/components/headers";
import { browseCrumbs } from "../crumbs";
import { browseAddAction } from "../actions";
import { LocalDay } from "../items/local-day";

export const metadata: Metadata = { title: "Podcast" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** A podcast's own page in Reflect, to explore before keeping it: its episodes, each heard here straight away. */
export default async function ShowPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const url = one(sp.u);
  if (!/^https:\/\/podcasts\.apple\.com\/.+\/id\d+/.test(url)) notFound();
  const [show, items] = await Promise.all([showFor(url), listItems(user.id)]);
  const kept = items.find((i) => i.medium === "podcast" && i.url === url);
  const units = kept ? await listUnits(user.id, kept.id) : [];
  const heard = new Set(units.filter((u) => u.doneOn).map((u) => u.unit));
  const title = one(sp.t) || show?.title || "Podcast";
  const by = one(sp.by);
  const image = one(sp.i);
  const episodeHref = (key: string) => (kept ? `/learn/items/${kept.id}/episodes/${key}` : `/learn/podcasts/episode?u=${encodeURIComponent(url)}&k=${key}&t=${encodeURIComponent(title)}`);
  return (
    <main className={PAGE}>
      <ThingHeader back="/learn/browse?m=podcast" crumbs={browseCrumbs("podcast")} title={title} />
      <div className="mt-6 space-y-6">
        <header className="flex gap-4">
          {/^https:\/\//.test(image) ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={image} alt="" className="size-28 shrink-0 rounded-lg border object-cover" />
          ) : null}
          <div className="min-w-0 space-y-2">
            <h1 id="page-title" className="text-2xl font-semibold tracking-tight">{title}</h1>
            {by ? <p className="text-muted-foreground">{by}</p> : null}
            <p className="text-muted-foreground text-sm">{show ? `${show.episodes.length} episodes, each heard here` : "The show's feed could not be read just now."}</p>
            {kept ? <Button asChild size="sm" variant="outline"><Link href={`/learn/items/${kept.id}`}>On your shelves</Link></Button> : (
              <form action={browseAddAction} className="flex flex-wrap gap-2">
                <input type="hidden" name="medium" value="podcast" /><input type="hidden" name="title" value={title} /><input type="hidden" name="url" value={url} />
                {by ? <input type="hidden" name="by" value={by} /> : null}{/^https:\/\//.test(image) ? <input type="hidden" name="image" value={image} /> : null}<LocalDay />
                <Button size="sm" name="go" value="start">Keep and listen</Button>
                <Button size="sm" variant="outline">Want to</Button>
              </form>
            )}
          </div>
        </header>
        {show?.episodes.length ? (
          <ol className="divide-y rounded-lg border px-4 text-sm">
            {show.episodes.slice(0, 100).map((e) => (
              <li key={e.key} className="space-y-1 py-3">
                <div className="flex items-center gap-3">
                  <Link href={episodeHref(e.key)} className="min-w-0 flex-1 font-medium underline-offset-4 hover:underline">{e.title}</Link>
                  <span className="text-muted-foreground shrink-0 text-xs tabular-nums">{heard.has(e.key) ? "Heard" : [e.date, e.seconds ? clock(e.seconds) : null].filter(Boolean).join(" · ")}</span>
                </div>
                {e.about ? <p className="text-muted-foreground line-clamp-2">{e.about}</p> : null}
              </li>
            ))}
          </ol>
        ) : null}
        <p className="text-muted-foreground text-xs">Played from the show&apos;s own feed. Reflect keeps no copy.</p>
      </div>
    </main>
  );
}
