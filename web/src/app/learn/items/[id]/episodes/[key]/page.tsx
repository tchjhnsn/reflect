import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { clock } from "@reflect/core/media";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { getItem, listUnits } from "@/server/ways-store";
import { showFor, transcriptOf } from "@/server/podcasts";
import { TaskHeader } from "@/components/headers";
import { UnitPlayer } from "../../../unit-player";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

/** One episode of a podcast, heard inside Reflect from the show's own feed, with its transcript when it has one. */
export default async function EpisodePage({ params }: { params: Promise<{ id: string; key: string }> }) {
  const user = await requireUser();
  const { id, key } = await params;
  const item = await getItem(user.id, id);
  if (!item) notFound();
  const show = await showFor(item.url);
  const at = show?.episodes.findIndex((e) => e.key === key) ?? -1;
  if (!show || at < 0) notFound();
  const e = show.episodes[at]!;
  const [transcript, units] = await Promise.all([transcriptOf(e), listUnits(user.id, id)]);
  const unit = units.find((u) => u.unit === key);
  const older = show.episodes[at + 1];
  return (
    <main className={PAGE}>
      <TaskHeader back={{ href: `/learn/items/${id}`, label: "Back to the show" }} title={item.title} position={[e.date, e.seconds ? clock(e.seconds) : ""].filter(Boolean).join(" · ") || undefined} />
      <div className="mt-6 space-y-5">
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">{e.title}</h1>
          {item.by ? <p className="text-muted-foreground">{item.by}</p> : null}
          {e.about ? <p className="max-w-prose whitespace-pre-line text-sm">{e.about}</p> : null}
        </header>
        <UnitPlayer
          item={id} unit={key} title={e.title} by={item.by} cite={[item.by, `${item.title}, “${e.title}”`].filter(Boolean).join(", ")}
          youtube={null} media={{ src: e.audio, kind: "audio" }} place={unit?.place ?? null} done={Boolean(unit?.doneOn)} of={null}
          transcript={transcript} notes={item.notes.filter((x) => x.unit === key)} next={older ? `/learn/items/${id}/episodes/${older.key}` : null}
        />
        <p className="text-muted-foreground text-xs">Played from the show&apos;s own feed. Reflect keeps no copy.</p>
      </div>
    </main>
  );
}
