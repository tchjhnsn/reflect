import Link from "next/link";
import { BookOpenIcon, PenLineIcon } from "lucide-react";
import { MEDIA } from "@reflect/core/ways";
import { Button } from "@/components/ui/button";
import { lifeActs, type LifeRecord } from "@/server/life-store";
import { listItems } from "@/server/ways-store";
import { getWork } from "@/server/work-store";
import { learningTowardAction, writeTowardAction } from "./space-actions";

const MEDIUM: Record<string, string> = Object.fromEntries(MEDIA.map((m) => [m.id, m.label]));

/**
 * Toward work the person wants: what they wrote about what moves them toward it (journal entries kept on it), and
 * what they are learning in Learn toward it, which is where Learn feeds Work.
 */
export async function Toward({ userId, record, back }: { userId: string; record: LifeRecord; back: string }) {
  const [acts, items] = await Promise.all([lifeActs(userId, record.id), listItems(userId)]);
  const entries = (await Promise.all(acts.filter((a) => a.kind === "wrote" && a.note).map(async (a) => ({ a, w: await getWork(userId, a.note!) })))).filter((x) => x.w);
  const chosen = new Set(typeof record.body.learning === "string" && record.body.learning ? record.body.learning.split(",") : []);
  const offered = items.filter((i) => chosen.has(i.id) || i.status !== "done" || i.subjects.length);
  const title = String(record.kind === "imagined" ? record.body.name ?? "" : record.body.title ?? "");
  const first = (body: string) => body.replace(/^>.*\n+/, "").trim().split("\n")[0]?.replace(/^#+\s*/, "").slice(0, 120) || "Not written yet";
  return (
    <section id="toward" className="grid scroll-mt-20 gap-6 rounded-xl border p-5">
      <div className="grid gap-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">What moves you toward it</h2>
            <p className="text-muted-foreground text-sm">Journal entries about what brought you closer to {title}.</p>
          </div>
          <form action={writeTowardAction.bind(null, record.id)}><Button type="submit" size="sm"><PenLineIcon />Write an entry</Button></form>
        </div>
        {entries.length ? (
          <ul className="divide-y rounded-lg border text-sm">
            {entries.map(({ a, w }) => (
              <li key={a.id}><Link href={`/journal/w/${w!.id}`} className="hover:bg-accent flex items-baseline justify-between gap-3 px-4 py-2"><span className="min-w-0 truncate">{w!.title || first(w!.body)}</span><span className="text-muted-foreground shrink-0 text-xs">{a.day}</span></Link></li>
            ))}
          </ul>
        ) : null}
      </div>
      <div className="grid gap-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Learning toward it</h2>
            <p className="text-muted-foreground text-sm">What you are learning or want to learn in Learn that is for this work.</p>
          </div>
          <Button asChild size="sm" variant="outline"><Link href={`/learn/browse?q=${encodeURIComponent(title)}`}><BookOpenIcon />Find something to learn</Link></Button>
        </div>
        {offered.length ? (
          <form action={learningTowardAction.bind(null, record.id, back)} className="grid gap-3">
            <div className="flex flex-wrap gap-2">
              {offered.slice(0, 40).map((i) => (
                <label key={i.id} className="has-[:checked]:bg-foreground has-[:checked]:text-background cursor-pointer rounded-full border px-3 py-1 text-sm">
                  <input type="checkbox" name="learning" value={i.id} defaultChecked={chosen.has(i.id)} className="sr-only" />
                  <span className="opacity-60">{MEDIUM[i.medium] ?? i.medium}{i.status === "doing" ? ", learning now" : i.status === "want" ? ", want to learn" : ", learned"}: </span>{i.title}
                </label>
              ))}
            </div>
            <div><Button type="submit" size="sm" variant="outline">Save</Button></div>
          </form>
        ) : <p className="text-muted-foreground text-sm">Nothing in Learn yet. What you add there can be marked toward this work.</p>}
      </div>
    </section>
  );
}
