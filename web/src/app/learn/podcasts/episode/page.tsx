import type { Metadata } from "next";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { clock } from "@reflect/core/media";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { listItems } from "@/server/ways-store";
import { showFor, transcriptOf } from "@/server/podcasts";
import { LearnHeader } from "../../header";
import { browseAddAction } from "../../actions";
import { LocalDay } from "../../items/local-day";
import { UnitPlayer } from "../../items/unit-player";

export const metadata: Metadata = { title: "Episode" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** An episode of a show not yet on your shelves, heard here; keeping the show keeps your place. */
export default async function PreviewEpisodePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const url = one(sp.u);
  const key = one(sp.k);
  if (!/^https:\/\/podcasts\.apple\.com\/.+\/id\d+/.test(url)) notFound();
  const kept = (await listItems(user.id)).find((i) => i.medium === "podcast" && i.url === url);
  if (kept) redirect(`/learn/items/${kept.id}/episodes/${key}`);
  const show = await showFor(url);
  const e = show?.episodes.find((x) => x.key === key);
  if (!show || !e) notFound();
  const title = one(sp.t) || show.title;
  const transcript = await transcriptOf(e);
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-5">
        <p className="text-muted-foreground text-sm"><Link href={`/learn/podcasts?u=${encodeURIComponent(url)}&t=${encodeURIComponent(title)}`} className="hover:underline">{title}</Link>{e.date ? ` · ${e.date}` : ""}{e.seconds ? ` · ${clock(e.seconds)}` : ""}</p>
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">{e.title}</h1>
          {e.about ? <p className="max-w-prose whitespace-pre-line text-sm">{e.about}</p> : null}
        </header>
        <form action={browseAddAction} className="bg-muted/50 flex flex-wrap items-center gap-2 rounded-lg border p-3 text-sm">
          <span className="min-w-0 flex-1">Keep the show to keep your place in each episode and take notes.</span>
          <input type="hidden" name="medium" value="podcast" /><input type="hidden" name="title" value={title} /><input type="hidden" name="url" value={url} /><LocalDay />
          <Button size="sm" name="go" value="start">Keep the show</Button>
        </form>
        <UnitPlayer item={null} unit={key} title={e.title} by={null} cite={`${title}, “${e.title}”`} youtube={null} media={{ src: e.audio, kind: "audio" }} place={null} done={false} of={null} transcript={transcript} notes={[]} next={null} />
      </div>
    </main>
  );
}
