import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { daily } from "@reflect/core/work-browse";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { lifeRecords } from "@/server/life-store";
import { CREDIT, fictionWork, wikiById } from "@/server/work-catalog";
import { Credit, EntryTop, Inspires, inspirationsOf, Part, WikiCard } from "../../parts";

export const dynamic = "force-dynamic";
export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  return { title: wikiById((await params).id)?.name ?? "Work" };
}

/** Work imagined in a story: a job or a company from a book, film or game. */
export default async function ImaginedWorkPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const w = wikiById(id);
  if (!w || w.kind === "past") notFound();
  const user = await requireUser();
  const kept = inspirationsOf(await lifeRecords(user.id, "work")).get(`fiction:${id}`) ?? null;
  const more = daily(fictionWork().filter((x) => x.id !== id), 4, id, "more");
  return (
    <main className={PAGE}>
      <EntryTop kind="Imagined in fiction" kindHref="/life/work/browse?m=imagined" title={w.name}>{w.from ? <p className="text-muted-foreground text-sm">From {w.from}</p> : null}</EntryTop>
      <div className="mt-6 grid max-w-4xl grid-cols-[minmax(0,1fr)] gap-8">
        {w.image ? <img src={`${w.image}?width=960`} alt="" className="bg-muted max-h-96 w-full rounded-xl object-contain" /> : null}
        {w.about ? <p className="text-base leading-relaxed first-letter:uppercase">{w.about}.</p> : null}
        <Inspires target={`fiction:${id}`} title={w.name} kept={kept} />
        {more.length ? <Part title="Other imagined work"><ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{more.map((x) => <WikiCard key={x.id} w={x} />)}</ul></Part> : null}
        <Credit>From <a href={`https://www.wikidata.org/wiki/${id}`} className="underline underline-offset-2">{CREDIT.wikidata.name}</a>.</Credit>
      </div>
    </main>
  );
}
